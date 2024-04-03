import concurrent.futures
import json
import random
import string
import time
import traceback
from collections import defaultdict
from copy import deepcopy
from typing import List

import pglast
import requests
from pglast import parse_sql
from pglast.ast import *
from pglast.enums.parsenodes import A_Expr_Kind
from pglast.enums.primnodes import BoolExprType, CoercionForm
from pglast.stream import RawStream
from pglast.visitors import Ancestor, Visitor
from psycopg2 import Error as psyconpg2Error
from sympy import Symbol, symbols
from sympy.logic.boolalg import And, Not, Or, to_dnf

from suql.postgresql_connection import execute_sql, execute_sql_with_column_info
from suql.prompt_continuation import llm_generate
from suql.utils import num_tokens_from_string

# System parameters, do not modify
SET_FREE_TEXT_FCNS = ["answer"]
verified_res = {}

# Embedding server address
EMBEDDING_SERVER_ADDRESS = "http://127.0.0.1:8509"
# This denotes the model used by the SUQL compiler for all verification purposes
MODEL = "gpt-3.5-turbo-0613"
# This denotes whether to attempt verify against all columns
# TODO: not yet used now
VERIFY_ALL_COLUMNS = True
# This denotes the max number of items per LIMIT number to be sent to GPT for verification
MAX_VERIFY = 20

# A list of declared tuples with their unique IDs
# e.g. TABLE_W_IDS = {"courses": "course_id", "ratings": "rating_id"}
# e.g. TABLE_W_IDS = {"restaurants": "_id"}
TABLE_W_IDS = {"restaurants": "_id"}


def generate_random_string(length=12):
    characters = string.ascii_lowercase + string.digits
    random_string = "".join(random.choice(characters) for _ in range(length))
    return random_string


class FreeTextFcnVisitor(Visitor):
    def __init__(self) -> None:
        super().__init__()
        self.set_free_text_fcns = SET_FREE_TEXT_FCNS
        self.res = False

    def __call__(self, node):
        super().__call__(node)

    def visit_FuncCall(self, ancestors, node: pglast.ast.FuncCall):
        for i in node.funcname:
            if i.sval in self.set_free_text_fcns:
                self.res = True
                return


def if_contains_free_text_fcn(node):
    visitor = FreeTextFcnVisitor()
    visitor(node)
    return visitor.res


class TypeCastAnswer(Visitor):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, node):
        super().__call__(node)

    def visit_TypeCast(self, ancestors, node: TypeCast):
        if (
            isinstance(node.arg, FuncCall)
            and isinstance(node.arg.funcname[0], String)
            and node.arg.funcname[0].sval == "answer"
        ):
            # if answer does not have exactly 2 paramters
            # it means that someone already filled the type in
            # then skip
            if len(node.arg.args) == 2:
                type_name = node.typeName.names[-1].sval
                node.arg.args = tuple(
                    list(node.arg.args) + [A_Const(val=String(sval=type_name))]
                )


class IfAllStructural(Visitor):
    def __init__(self) -> None:
        super().__init__()
        self.res = True

    def __call__(self, node):
        super().__call__(node)

    def visit_A_Expr(self, ancestors, node: A_Expr):
        if self.res is False:
            return

        def is_structural(expr):
            if (
                isinstance(expr, FuncCall)
                and ".".join(map(lambda x: x.sval, expr.funcname)) in SET_FREE_TEXT_FCNS
            ):
                return False
            return True

        if not (is_structural(node.lexpr) and is_structural(node.rexpr)):
            self.res = False


def get_sublink_parent(ancestor):
    if ancestor.node is not None and not isinstance(ancestor.node, tuple):
        return ancestor.node
    return get_sublink_parent(ancestor.parent)


class IfInvovlesSubquery(Visitor):
    def __init__(self) -> None:
        super().__init__()
        self.sublinks = []
        self.ancestors = []

    def __call__(self, node):
        super().__call__(node)

    def visit_SubLink(self, ancestors, node: SubLink):
        # note: it's possible that the subqueries are parallel to each other
        # in this case we need to return a list of subqueries, but we would only return the ones
        # on the same level

        # hiearchy-based recursion would take care of nested SubLinks
        self.sublinks.append(node.subselect)
        self.ancestors.append(ancestors)

    def return_top_level_sublinks(self):
        if len(self.sublinks) == 1:
            return self.sublinks

        # the purpose of this function is to return only the top sublinks
        res = []
        closest_parent = None
        for sublink, ancestor in zip(self.sublinks, self.ancestors):
            # first get the actual parent node
            if ancestor.parent is None:
                continue
            else:
                closest_parent = get_sublink_parent(ancestor.parent)
            if closest_parent == closest_parent:
                res.append(sublink)
        return res


def if_all_structural(node):
    visitor = IfAllStructural()
    visitor(node)
    return visitor.res


class SelectVisitor(Visitor):
    def __init__(
        self,
        fts_fields,
        embedding_server_address,
        select_username,
        select_userpswd,
        create_username,
        create_userpswd,
    ) -> None:
        super().__init__()
        self.tmp_tables = []
        # this cache is to store classifier results for sturctured fields (e.g. cuisines in restaurants)
        self.cache = defaultdict(dict)
        self.fts_fields = fts_fields
        self.embedding_server_address = embedding_server_address

        # stores credentials
        self.select_username = select_username
        self.select_userpswd = select_userpswd
        self.create_username = create_username
        self.create_userpswd = create_userpswd

    def __call__(self, node):
        super().__call__(node)

    def visit_SelectStmt(self, ancestors, node: SelectStmt):
        type_cast_answer_visitor = TypeCastAnswer()
        type_cast_answer_visitor(node)

        if not node.whereClause:
            return

        # First, understand whether this involves subquery. If it does, then starts with that first and builds upwards
        # If that subquery in turn involves other subqueries, then recursive calls take care of it
        subquery_visitor = IfInvovlesSubquery()
        subquery_visitor(node)
        sublinks = subquery_visitor.return_top_level_sublinks()
        for sublink in sublinks:
            self.visit_SelectStmt(None, sublink)

        freeTextFcnVisitor = FreeTextFcnVisitor()
        freeTextFcnVisitor(node.whereClause)

        if freeTextFcnVisitor.res:
            tmp_table_name = "temp_table_{}".format(generate_random_string())
            self.tmp_tables.append(tmp_table_name)

            # main entry point for SUQL compiler optimization
            results, column_info = analyze_SelectStmt(
                node,
                self.cache,
                self.fts_fields,
                self.embedding_server_address,
                self.select_username,
                self.select_userpswd,
            )

            # based on results and column_info, insert a temporary table
            column_create_stmt = ",\n".join(
                list(map(lambda x: f'"{x[0]}" {x[1]}', column_info))
            )
            create_stmt = f"CREATE TABLE {tmp_table_name} (\n{column_create_stmt}\n); GRANT SELECT ON {tmp_table_name} TO {self.select_username};"
            print("created table {}".format(tmp_table_name))
            execute_sql(
                create_stmt,
                user=self.create_username,
                password=self.create_userpswd,
                commit_in_lieu_fetch=True,
                no_print=True,
            )

            if results:
                # some special processing is needed for python dict types - they need to be converted to json
                json_indices = [
                    index
                    for index, element in enumerate(column_info)
                    if element[1] in ("json", "jsonb")
                ]
                placeholder_str = ", ".join(["%s"] * len(results[0]))
                for result in results:
                    updated_results = tuple(
                        [
                            json.dumps(element) if index in json_indices else element
                            for index, element in enumerate(result)
                        ]
                    )
                    execute_sql(
                        f"INSERT INTO {tmp_table_name} VALUES ({placeholder_str})",
                        data=updated_results,
                        user=self.create_username,
                        password=self.create_userpswd,
                        commit_in_lieu_fetch=True,
                    )

            # finally, modify the existing sql with tmp_table_name
            node.fromClause = (
                RangeVar(relname=tmp_table_name, inh=True, relpersistence="p"),
            )
            node.whereClause = None
        else:
            classify_db_fields(
                node,
                self.cache,
                self.fts_fields,
                self.select_username,
                self.select_userpswd,
            )

    def serialize_cache(self):
        def print_value(x):
            if isinstance(x, String):
                return x.sval
            elif isinstance(x, Integer):
                return x.ival
            elif isinstance(x, Float):
                return x.fval
            else:
                raise ValueError()

        res = deepcopy(self.cache)
        for i in res:
            for j in res[i]:
                if isinstance(res[i][j], tuple):
                    res[i][j] = list(map(lambda x: print_value(x), res[i][j]))
                # special case originally for HybridQA
                # this denotes that a predicate should be thrown away
                elif isinstance(res[i][j], bool):
                    res[i][j] = [res[i][j]]
                else:
                    res[i][j] = [res[i][j].sval]
        return dict(res)

    def drop_tmp_tables(self):
        for tmp_table_name in self.tmp_tables:
            drop_stmt = f"DROP TABLE {tmp_table_name}"
            execute_sql(
                drop_stmt,
                user=self.create_username,
                password=self.create_userpswd,
                commit_in_lieu_fetch=True,
            )


class PredicateMapping:
    def __init__(self) -> None:
        self.symbols2predicate = {}
        self.counter = 0

    def add_mapping(self, predicate: BoolExpr or A_Expr) -> Symbol:
        res = symbols(str(self.counter))
        self.symbols2predicate[res] = predicate
        self.counter += 1
        return res

    def retrieve_predicate(self, symbol):
        return self.symbols2predicate[symbol]


def convert2dnf(predicate):
    predicate_mapping = PredicateMapping()

    def predicate2symbol(_predicate: BoolExpr):
        if if_all_structural(_predicate):
            return predicate_mapping.add_mapping(_predicate)

        if isinstance(_predicate, A_Expr):
            return predicate_mapping.add_mapping(_predicate)
        if _predicate.boolop == BoolExprType.AND_EXPR:
            return And(*(predicate2symbol(x) for x in _predicate.args))
        if _predicate.boolop == BoolExprType.OR_EXPR:
            return Or(*(predicate2symbol(x) for x in _predicate.args))
        if _predicate.boolop == BoolExprType.NOT_EXPR:
            return Not(*(predicate2symbol(x) for x in _predicate.args))
        else:
            raise ValueError()

    def symbol2predicate(symbol_predicate):
        if isinstance(symbol_predicate, And):
            return BoolExpr(
                boolop=BoolExprType.AND_EXPR,
                args=tuple(symbol2predicate(arg) for arg in symbol_predicate.args),
            )
        if isinstance(symbol_predicate, Or):
            return BoolExpr(
                boolop=BoolExprType.OR_EXPR,
                args=tuple(symbol2predicate(arg) for arg in symbol_predicate.args),
            )
        if isinstance(symbol_predicate, Not):
            return BoolExpr(
                boolop=BoolExprType.NOT_EXPR,
                args=tuple(symbol2predicate(arg) for arg in symbol_predicate.args),
            )
        else:
            return predicate_mapping.retrieve_predicate(symbol_predicate)

    if isinstance(predicate, A_Expr):
        return predicate

    elif isinstance(predicate, BoolExpr):
        symbol_predicate = predicate2symbol(predicate)
        dnf_symbol_predicate = to_dnf(symbol_predicate)
        sql_expr = symbol2predicate(dnf_symbol_predicate)
        return sql_expr


def verify(document, field, query, operator, value):
    if (document, field, query, operator, value) in verified_res:
        return verified_res[(document, field, query, operator, value)]

    # construct the answer part
    if operator == "=":
        answer = value
    else:
        answer = operator + " " + value

    res = llm_generate(
        template_file="prompts/verification.prompt",
        prompt_parameter_values={
            "document": document,
            "field": field[1],  # field is a tuple (table_name, field_name)
            "query": query,
            "answer": answer,
        },
        engine=MODEL,
        temperature=0,
        stop_tokens=["\n"],
        max_tokens=30,
        postprocess=False,
    )[0]

    if "the answer is correct" in res.lower():
        res = True
    else:
        res = False
    verified_res[(document, field, query, operator, value)] = res
    return res


def verify_single_res(doc, field_query_list):
    # verify for each stmt, if any stmt fails to verify, exclude it
    all_found = True
    found_stmt = []
    for i, entry in enumerate(field_query_list):
        field, query, operator, value = entry

        # this function helps with list processing
        # for a list, verify against each one, if any returns true then return that
        def verify_single_value(single_value, single_column_name):
            res = False
            # if this is a string, then directly verify:
            if isinstance(single_value, str):
                res = verify(single_value, single_column_name, query, operator, value)
            # otherwise it is a list. Go over the list until if one verifies
            else:
                assert isinstance(single_value, list)
                for current_column_each_value in single_value:
                    if verify(
                        current_column_each_value,
                        single_column_name,
                        query,
                        operator,
                        value,
                    ):
                        res = True
                        break
            return res

        # if there are three elements in doc, then this indicates that we are in the mode
        # of attempting to verify all columns
        # the other columns do not have an inherent order for now
        if len(doc) == 3:
            # first, try to find the results in the current column:
            current_column_verified = verify_single_value(doc[1][i], field)

            if current_column_verified:
                all_found = True
            else:
                # if results can't be verified with the current column
                # then set all_found = False
                all_found = False

                # iterate all other columns, attempt to find one, if found then revert all_found
                for other_column, other_value in doc[2]:
                    if other_column == field:
                        continue

                    other_column_verified = verify_single_value(
                        other_value, other_column
                    )
                    if other_column_verified:
                        all_found = True
                        break

                # if still can't find an answer, then this row is hopeless and should be rejected
                if not all_found:
                    break

        else:
            if not verify(doc[1][i], field, query, operator, value):
                all_found = False
                break
            else:
                found_stmt.append(
                    "Verified answer({}, '{}') {} {} in table = {} based on document: {}".format(
                        field[1], query, operator, value, field[0], doc[1][i]
                    )
                )
    if all_found:
        print("\n".join(found_stmt))
    elif found_stmt:
        print("partially verified: " + "\n".join(found_stmt))

    return all_found


def parallel_filtering(fcn, source: list, limit, enforce_ordering=False):
    true_count = 0
    true_items = set()

    # build a dictionary with index -> true/false/None
    # which indicates whether an item has been verified
    ordered_results = {i: None for i in range(len(source))}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(fcn, item): item for item in source}

        for future in concurrent.futures.as_completed(futures):
            item = futures[future]
            result = future.result()
            if result:
                true_count += 1
                if isinstance(item[0], list):
                    true_items.update(item[0])
                else:
                    true_items.add(item[0])
                ordered_results[source.index(item)] = True
            else:
                ordered_results[source.index(item)] = False

            # TODO: for best performance, if enforce_ordering = True,
            # cancel remaining futures if enough top results have been found

            if true_count >= limit and limit != -1 and not enforce_ordering:
                # Cancel remaining futures
                for f in futures:
                    f.cancel()
                break

    if enforce_ordering:
        res = []
        for i, item in enumerate(source):
            if ordered_results[i]:
                if isinstance(item[0], list):
                    res += item[0]
                else:
                    res.append(item[0])
        return res

    return true_items


def retrieve_and_verify(
    node: SelectStmt,
    field_query_list,
    existing_results,
    column_info,
    limit,
    embedding_server_address,
    parallel=True,
    fetch_all=False,
):
    # field_query_list is a list of tuples, each entry of the tuple are:
    # 0st: field: field to do retrieval on
    # 1st: query: query for retrieval model
    # 2nd: operator: operator to compare against
    # 3rd: value: value to compare against
    #
    # existing_results: existing results to run retrieval on, this is a list of tuples
    # column_info: this is a list of tuples, first element is column name and second is type
    # limit: max number of returned results
    if len(node.fromClause) == 1 and isinstance(node.fromClause[0], RangeVar):
        id_field_name = TABLE_W_IDS[node.fromClause[0].relname]
        single_table = True
        id_index = list(map(lambda x: x[0], column_info)).index(id_field_name)
        id_list = list(map(lambda x: x[id_index], existing_results))
    elif len(node.fromClause) == 1 and isinstance(node.fromClause[0], JoinExpr):
        single_table = False
        id_list = {}
        for arg in [node.fromClause[0].larg, node.fromClause[0].rarg]:
            if isinstance(arg, RangeVar):
                table_name = arg.relname
                id_field_name = TABLE_W_IDS[table_name]
                id_index = list(map(lambda x: x[0], column_info)).index(
                    f"{table_name}^{id_field_name}"
                )
                id_list[table_name] = list(map(lambda x: x[id_index], existing_results))
        # also append a special "_id_join" field to keep track
        column_info = [("_id_join", "int4")] + column_info
        _id_join_index = list(map(lambda x: x[0], column_info)).index("_id_join")
        existing_results = [(i,) + v for i, v in enumerate(existing_results)]
        id_index = 0
        id_list["_id_join"] = list(map(lambda x: x[_id_join_index], existing_results))
    elif len(node.fromClause) > 1 and isinstance(node.fromClause, tuple):
        # this is a case with self joins, create a new _id column
        # TODO: from hybridQA. Check whether this works with retriever
        single_table = False
        existing_results = [(i,) + v for i, v in enumerate(existing_results)]
        column_info = [("_id_join", "int4")] + column_info
        id_index = list(map(lambda x: x[0], column_info)).index("_id_join")
        id_list = list(map(lambda x: x[id_index], existing_results))
    else:
        # some yet-to-be-discovered cases?
        raise ValueError()

    start_time = time.time()
    # clean previous verified_res, to avoid determistic caching (this could be removed)
    global verified_res
    verified_res = {}

    if fetch_all:
        # first get all free text fields:
        all_free_text_columns = []
        # NOTE: uncomment the following to enable going to multiple columns for filtering
        for each_column_info in column_info:
            # NOTE: HybridQA-specific
            if each_column_info[1] == "text[]" and each_column_info[0].endswith(
                "_Info"
            ):
                all_free_text_columns.append(each_column_info[0])

        # this is a list of list of tuples of the form
        # [1, [(field_name, field_value), ...], ...]

        parsed_result = []
        for existing_res in existing_results:
            intermediate_result = []
            other_columns_intermediate_result = []
            for field in field_query_list:
                field_index = list(map(lambda x: x[0], column_info)).index(field[0])
                intermediate_result.append(existing_res[field_index])

            for other_column in all_free_text_columns:
                other_columns_intermediate_result.append(
                    (
                        other_column,
                        existing_res[
                            list(map(lambda x: x[0], column_info)).index(other_column)
                        ],
                    )
                )

            parsed_result.append(
                [
                    existing_res[id_index],
                    intermediate_result,
                    other_columns_intermediate_result,
                ]
            )
    else:
        start_time = time.time()
        data = {
            "id_list": id_list,
            "field_query_list": list(map(lambda x: (x[0], x[1]), field_query_list)),
            "top": limit
            * MAX_VERIFY,  # return MAX_VERIFY times the ordered amount, for GPT filtering purposes
        }
        data["single_table"] = single_table

        # Send a POST request
        response = requests.post(
            embedding_server_address + "/search",
            json=data,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        parsed_result = response.json()["result"]

        filtered_parsed_result = []
        for res in parsed_result:
            if res not in filtered_parsed_result:
                filtered_parsed_result.append(res)
        parsed_result = filtered_parsed_result

    if parallel:
        # parallelize verification calls
        id_res = parallel_filtering(
            lambda x: verify_single_res(x, field_query_list),
            parsed_result,
            limit,
            enforce_ordering=True if node.sortClause is not None else False,
        )
    else:
        id_res = []
        for each_res in parsed_result:
            if verify_single_res(each_res, field_query_list):
                id_res.append(each_res[0])

    end_time = time.time()
    print("retrieve + verification time {}s".format(end_time - start_time))

    if single_table:
        res = list(filter(lambda x: x[id_index] in id_res, existing_results))
    else:
        res = [
            i[1:]
            for i in list(filter(lambda x: x[id_index] in id_res, existing_results))
        ]
    return res


def get_a_expr_field_value(node: A_Expr, no_check=False):
    if isinstance(node.lexpr, ColumnRef):
        column = node.lexpr
        value = node.rexpr
    elif isinstance(node.rexpr, ColumnRef):
        column = node.rexpr
        value = node.lexpr
    elif no_check:
        return None, None

    if isinstance(column, ColumnRef) and isinstance(value, A_Const):
        column_name = column.fields[0].sval
        value_res = value.val
        return column_name, value_res
    elif no_check:
        return None, None
    else:
        raise ValueError()


def replace_a_expr_field(node: A_Expr, ancestors: Ancestor, new_value):
    def find_value_column(node: A_Expr):
        if isinstance(node.lexpr, ColumnRef):
            column = node.lexpr
            value = node.rexpr
        else:
            column = node.rexpr
            value = node.lexpr

        assert isinstance(column, ColumnRef)
        assert isinstance(value, A_Const)
        return column, value

    if isinstance(new_value, tuple):
        res = []
        for i in new_value:
            new_atomic_value = deepcopy(node)
            _, value = find_value_column(new_atomic_value)
            value.val = i
            res.append(new_atomic_value)
        new_node = BoolExpr(BoolExprType.OR_EXPR, args=tuple(res))

        # case where the parent is directly a select stmt
        if isinstance(ancestors.node, SelectStmt):
            ancestors.node.whereClause = new_node
        # case where the parent is a boolean expression, which means we need to replace the respective one
        if isinstance(ancestors.parent.node, BoolExpr):
            new_args = list(ancestors.parent.node.args)
            new_args[ancestors.member] = new_node
            ancestors.parent.node.args = tuple(new_args)
    # originally for HybridQA: delete predicate if this predicate returns no results
    elif new_value is True:
        # case where the parent is directly a select stmt
        if isinstance(ancestors.node, SelectStmt):
            ancestors.node.whereClause = None
        # case where the parent is a boolean expression. Delete this node
        elif isinstance(ancestors.parent.node, BoolExpr):
            new_args = list(ancestors.parent.node.args)
            new_args = [x for x in new_args if x != node]
            ancestors.parent.node.args = tuple(new_args)
    else:
        _, value = find_value_column(node)
        value.val = new_value


def greedy_search_comma(input_string, predefined_list):
    chunks = input_string.split(",")
    results = []
    buffer = ""

    for chunk in chunks:
        buffer = buffer + chunk if buffer else chunk

        if buffer.strip() in predefined_list:
            results.append(buffer.strip())
            buffer = ""
        elif buffer in predefined_list:
            results.append(buffer.strip())
            buffer = ""
        else:
            buffer += ","

    # After iterating through all chunks, check if there's any remaining buffer
    if buffer and buffer.strip() in predefined_list:
        results.append(buffer.strip())

    return results


# NOTE: this function is used if an ENUM field contains commas
# in which case asking an LLM to output a comma-seperated list could be problematic
# currently unused
def get_comma_separated_numbers(input_string):
    try:
        # Remove spaces from the input string and then split it into a list of values using a comma as the separator
        values = input_string.replace(" ", "").split(",")

        # Convert each value to a float and store them in a list
        num_values = [int(value) for value in values]

        return num_values  # Return the list of numbers
    except ValueError:
        # If a ValueError is raised, it means at least one value is not a valid number, so return False
        return False


class StructuralClassification(Visitor):
    def __init__(
        self, node: SelectStmt, cache, fts_fields, select_username, select_userpswd
    ) -> None:
        super().__init__()
        self.node = node
        self.cache = cache
        self.fts_fields = fts_fields
        self.select_username = select_username
        self.select_userpswd = select_userpswd

    def __call__(self, node):
        super().__call__(node)

    def visit_A_Expr(self, ancestors: Ancestor, node: A_Expr):
        # TODO: handle containment clauses
        if isinstance(node.name[0], String) and node.name[0].sval in ["@>", "<@"]:
            parsed_res = True
            replace_a_expr_field(node, ancestors, parsed_res)
            return

        # if this appears in projection, disregard
        if ancestors.find_nearest(ResTarget) is not None:
            return
        assert if_all_structural(node)

        if isinstance(node.lexpr, SubLink) or isinstance(node.rexpr, SubLink):
            # if there is a subquery, skip classification
            return

        if isinstance(node.lexpr, ColumnRef) and isinstance(node.rexpr, ColumnRef):
            return

        if isinstance(node.name[0], String) and node.name[0].sval in ["BETWEEN"]:
            # skip certain numerical comparisons
            return

        skip_fcn_list = ["try_cast", "count"]
        if (
            isinstance(node.lexpr, FuncCall)
            and isinstance(node.lexpr.funcname[0], String)
            and node.lexpr.funcname[0].sval in skip_fcn_list
        ) or (
            isinstance(node.rexpr, FuncCall)
            and isinstance(node.rexpr.funcname[0], String)
            and node.rexpr.funcname[0].sval in skip_fcn_list
        ):
            # if it involves certain functions, skip
            return

        # if this is one of the fields declared to be used with fts (full text search), convert so:
        for table_name, field_name in self.fts_fields:
            if (
                len(self.node.fromClause) == 1
                and isinstance(self.node.fromClause[0], RangeVar)
                and node.name[0].sval in ["~~", "~~*", "="]
            ):
                n_field_name, n_value_name = get_a_expr_field_value(node, no_check=True)
                if (
                    table_name == self.node.fromClause[0].relname
                    and field_name == n_field_name
                ):
                    node.lexpr = FuncCall(
                        funcname=(String(sval="websearch_to_tsquery"),),
                        args=(
                            A_Const(
                                val=String(sval=n_value_name.sval.replace("%", ""))
                            ),
                        ),
                    )
                    node.name = (String(sval="@@"),)
                    node.kind = A_Expr_Kind.AEXPR_OP
                    node.rexpr = FuncCall(
                        funcname=(String(sval="to_tsvector"),),
                        args=(ColumnRef(fields=(String(sval=n_field_name),)),),
                    )
                    return

        to_execute_node = deepcopy(self.node)

        # change projection to include everything
        to_execute_node.targetList = (ResTarget(val=ColumnRef(fields=(A_Star(),))),)
        # set limit to 1, to see if there are results
        to_execute_node.limitCount = Integer(ival=1)
        # change predicates
        to_execute_node.whereClause = node
        # reset any groupby clause
        to_execute_node.groupClause = None
        # reset sorting
        to_execute_node.sortClause = None

        try:
            res, column_infos = execute_sql_with_column_info(
                RawStream()(to_execute_node),
                unprotected=True,
                user=self.select_username,
                password=self.select_userpswd,
            )
            # it is possible if there is a type error
            # e.g. "Passengers ( 2017 )" = '490,000', but "Passengers ( 2017 )" is actually of type int
            # in such cases, the `column_infos` variable would not capture the actual type and instead stores an empty list
            # thus execute an empty query just to find out the type
            if column_infos == []:
                to_execute_node.whereClause = None
                _, column_infos = execute_sql_with_column_info(
                    RawStream()(to_execute_node),
                    user=self.select_username,
                    password=self.select_userpswd,
                )
        except psyconpg2Error:
            print(
                "above error happens during ENUM classification attempts. Marking this predicate as returning answer."
            )
            res = True

        if not res:
            print("determined the above predicate returns no result")
            # try to classify into one of the known values
            # first, we need to find out what is the value here - some heuristics here to find out
            column_name, value_res = get_a_expr_field_value(node)

            if isinstance(value_res, String):
                value_res_clear = value_res.sval
            elif isinstance(value_res, Integer):
                value_res_clear = value_res.ival
            elif isinstance(value_res, Float):
                value_res_clear = value_res.fval
            else:
                raise ValueError()

            print(
                "determined column name: {}; value: {}".format(
                    column_name, value_res_clear
                )
            )

            # first check if this is already in the cache
            # TODO: for best performance move this before the execution above
            if column_name in self.cache and value_res_clear in self.cache[column_name]:
                replace_a_expr_field(
                    node, ancestors, self.cache[column_name][value_res_clear]
                )
            else:
                # first understand whether this field is of type TEXT or TEXT[]
                column_type = None
                for column_info in column_infos:
                    if column_info[0] == column_name:
                        column_type = column_info[1]
                        break

                assert column_type is not None
                if column_type.lower() == "text[]":
                    # the SQL for getting results from TEXT[] is a bit complicated
                    # suppose the field is called cuisines, then this is the desired SQL
                    # SELECT DISTINCT unnest(cuisines) FROM restaurants;
                    to_execute_node.targetList = (
                        ResTarget(
                            val=FuncCall(
                                agg_distinct=False,
                                agg_star=False,
                                agg_within_group=False,
                                func_variadic=False,
                                funcformat=CoercionForm.COERCE_EXPLICIT_CALL,
                                args=(
                                    ColumnRef(
                                        fields=(column_name,),
                                    ),
                                ),
                                funcname=(String(sval=("unnest")),),
                            )
                        ),
                    )
                    to_execute_node.distinctClause = (None,)
                else:
                    to_execute_node.targetList = (
                        ResTarget(val=ColumnRef(fields=(column_name,))),
                    )
                    # TODO: maybe None would also work
                    to_execute_node.distinctClause = (
                        ResTarget(val=ColumnRef(fields=(column_name,))),
                    )

                to_execute_node.sortClause = None
                to_execute_node.whereClause = None
                to_execute_node.limitCount = None  # find all entries
                field_value_choices, _ = execute_sql_with_column_info(
                    RawStream()(to_execute_node),
                    user=self.select_username,
                    password=self.select_userpswd,
                )
                # TODO deal with list problems?
                field_value_choices = list(map(lambda x: x[0], field_value_choices))
                field_value_choices.sort()

                # if a field has too many values (judged by tokens),
                # then it is not a enum field.
                # we give up on these (i.e., just use the existing predicate)
                if num_tokens_from_string("\n".join(field_value_choices)) <= 3000:
                    res = llm_generate(
                        "prompts/field_classification.prompt",
                        {
                            "predicted_field_value": value_res_clear,
                            "field_value_choices": field_value_choices,
                            "field_name": column_name,
                        },
                        engine=MODEL,
                        temperature=0,
                        stop_tokens=["\n"],
                        max_tokens=100,
                        postprocess=False,
                    )[0]
                    if res in field_value_choices:
                        replace_a_expr_field(node, ancestors, String(sval=(res)))
                        self.cache[column_name][value_res_clear] = String(sval=(res))
                    # tries to parse it as a list
                    else:
                        parsed_res = greedy_search_comma(res, field_value_choices)
                        if parsed_res:
                            parsed_res = tuple(
                                map(lambda x: String(sval=(x)), parsed_res)
                            )
                            replace_a_expr_field(node, ancestors, parsed_res)
                            self.cache[column_name][value_res_clear] = parsed_res


def classify_db_fields(
    node: SelectStmt,
    cache: dict,
    fts_fields: List,
    select_username: str,
    select_userpswd: str,
):
    # we expect all atomic predicates under `predicate` to only involve stru fields
    # (no `answer` function)
    # the goal of this function is to determine which predicate leads to no results
    # for a field without results, try to classify into one of the existing fields
    visitor = StructuralClassification(
        node, cache, fts_fields, select_username, select_userpswd
    )
    visitor(node)


class Replace_Original_Target_Visitor(Visitor):
    def __init__(self, table_column_mapping={}) -> None:
        super().__init__()
        self.table_column_mapping: dict = table_column_mapping

    def __call__(self, node):
        super().__call__(node)

    def visit_ColumnRef(self, ancestors: Ancestor, node: ColumnRef):
        if len(list(map(lambda x: x.sval, node.fields))) > 1:
            assert len(list(map(lambda x: x.sval, node.fields))) == 2
            node.fields = (String(sval="^".join(map(lambda x: x.sval, node.fields))),)
        else:
            res = None
            for table_name, columns in self.table_column_mapping.items():
                if node.fields[0].sval in map(lambda x: x[0], columns):
                    if res is not None:
                        # the same field appears twice, this means that the original syntax is problematic
                        break
                    res = (String(sval=f"{table_name}^{node.fields[0].sval}"),)
            node.fields = res


def execute_structural_sql(
    original_node: SelectStmt,
    predicate: BoolExpr,
    cache: dict,
    fts_fields: List,
    select_username: str,
    select_userpswd: str,
):
    node = deepcopy(original_node)
    # change projection to include everything
    # there are a couple of cases here
    # the simplest case is if it is one table
    if len(node.fromClause) == 1 and isinstance(node.fromClause[0], RangeVar):
        node.targetList = (ResTarget(val=ColumnRef(fields=(A_Star(),))),)
    # these are full joins
    elif len(node.fromClause) == 1 and isinstance(node.fromClause[0], JoinExpr):
        all_projection_fields = []
        table_column_mapping = {}
        for table in [node.fromClause[0].larg, node.fromClause[0].rarg]:
            # find out what columns this table has
            _, columns = execute_structural_sql(
                SelectStmt(fromClause=(table,)),
                None,
                cache,
                fts_fields,
                select_username,
                select_userpswd,
            )
            # give the projection fields new names
            projection_table_name = (
                table.alias.aliasname if table.alias is not None else table.relname
            )
            table_column_mapping[projection_table_name] = columns

            for column in columns:
                new_projection_clause = ResTarget(
                    name=f"{projection_table_name}^{column[0]}",
                    val=ColumnRef(
                        fields=(
                            String(sval=projection_table_name),
                            String(sval=column[0]),
                        )
                    ),
                )
                all_projection_fields.append(new_projection_clause)
        node.targetList = tuple(all_projection_fields)

        # if we replaced the names, we also need to propagate the names to the original `node` in projections
        replace_original_target_visitor = Replace_Original_Target_Visitor(
            table_column_mapping=table_column_mapping
        )
        replace_original_target_visitor(original_node.targetList)
        if original_node.sortClause is not None:
            replace_original_target_visitor(original_node.sortClause)
    # next, there are tuple joins (self joins)
    elif len(node.fromClause) > 1 and isinstance(node.fromClause, tuple):
        all_projection_fields = []
        for table in node.fromClause:
            # find out what columns this table has
            _, columns = execute_structural_sql(
                SelectStmt(fromClause=(table,)),
                None,
                cache,
                fts_fields,
                select_username,
                select_userpswd,
            )
            # give the projection fields new names
            projection_table_name = (
                table.alias.aliasname if table.alias is not None else table.relname
            )
            for column in columns:
                new_projection_clause = ResTarget(
                    name=f"{projection_table_name}^{column[0]}",
                    val=ColumnRef(
                        fields=(
                            String(sval=projection_table_name),
                            String(sval=column[0]),
                        )
                    ),
                )
                all_projection_fields.append(new_projection_clause)
        node.targetList = tuple(all_projection_fields)

        # if we replaced the names, we also need to propagate the names to the original `node` in projections
        replace_original_target_visitor = Replace_Original_Target_Visitor()
        replace_original_target_visitor(original_node.targetList)
        if original_node.sortClause is not None:
            replace_original_target_visitor(original_node.sortClause)

    # Some other cases?
    else:
        node.targetList = (ResTarget(val=ColumnRef(fields=(A_Star(),))),)

    # reset all limits
    node.limitCount = None
    node.limitOffset = None
    # change predicates
    node.whereClause = predicate
    node.groupClause = None
    node.havingClause = None

    # only queries that involve only structural parts can be executed
    assert if_all_structural(node)

    # deal with sturctural field classification
    classify_db_fields(node, cache, fts_fields, select_username, select_userpswd)

    sql = RawStream()(node)
    print("execute_structural_sql executing sql: {}".format(sql))
    return execute_sql_with_column_info(
        sql, user=select_username, password=select_userpswd
    )


def execute_free_text_queries(
    node,
    predicate: BoolExpr,
    existing_results,
    column_info,
    limit,
    embedding_server_address,
):
    # the predicate should only contain an atomic unstructural query
    # or an AND of multiple unstructural query (NOT of an unstructural query is considered to be atmoic)
    def assert_A_Const_Or_Tuple_A_Const(v):
        if isinstance(v, A_Const):
            return True

        # Assert that value is a tuple
        assert isinstance(v, tuple), "Value must be a tuple"

        # Assert that each element in the tuple is of type A_Const
        for element in v:
            assert isinstance(element, A_Const), "Each element must be of type A_Const"

        return True

    def extract_tuple_value(v: List[A_Const]):
        res = []
        for i in v:
            if isinstance(i.val, String):
                res.append(i.val.sval)
            elif isinstance(i.val, Integer):
                res.append(i.val.ival)
            else:
                raise ValueError()
        return tuple(res)

    def breakdown_unstructural_query(predicate: A_Expr):
        assert if_contains_free_text_fcn(predicate.lexpr) or if_contains_free_text_fcn(
            predicate.rexpr
        )
        if if_contains_free_text_fcn(predicate.lexpr) and if_contains_free_text_fcn(
            predicate.rexpr
        ):
            raise ValueError(
                "cannot optimize for predicate containing free text functions on both side of expression: {}".format(
                    RawStream()(predicate)
                )
            )

        free_text_clause = (
            predicate.lexpr
            if if_contains_free_text_fcn(predicate.lexpr)
            else predicate.rexpr
        )
        value_clause = (
            predicate.lexpr
            if not if_contains_free_text_fcn(predicate.lexpr)
            else predicate.rexpr
        )
        assert isinstance(free_text_clause, FuncCall)
        assert assert_A_Const_Or_Tuple_A_Const(value_clause)

        query_lst = list(
            filter(lambda x: isinstance(x, A_Const), free_text_clause.args)
        )
        assert len(query_lst) == 1
        query = query_lst[0].val.sval

        field_lst = list(
            filter(lambda x: isinstance(x, ColumnRef), free_text_clause.args)
        )
        assert len(field_lst) == 1

        field = tuple(map(lambda x: x.sval, field_lst[0].fields))
        if len(field) > 1:
            assert len(field) == 2
        else:
            # we need to find out the associated table for this field
            if len(node.fromClause) == 1 and isinstance(node.fromClause[0], RangeVar):
                field = (
                    node.fromClause[0].relname,
                    field_lst[0].fields[0].sval,
                )
            else:
                for column_name, _ in column_info:
                    if "^" not in column_name:
                        continue
                    else:
                        if column_name.split("^")[1] == field_lst[0].fields[0].sval:
                            field = tuple(column_name.split("^"))

        operator = predicate.name[0].sval
        if isinstance(value_clause, tuple):
            value = extract_tuple_value(value_clause)
        elif isinstance(value_clause.val, String):
            value = value_clause.val.sval
        elif isinstance(value_clause.val, Integer):
            value = value_clause.val.ival
        else:
            raise ValueError()

        return field, query, operator, value

    # TODO: handle cases with NOT

    if predicate is None:
        return existing_results
    if isinstance(predicate, A_Expr):
        field, query, operator, value = breakdown_unstructural_query(predicate)
        return (
            retrieve_and_verify(
                node,
                [(field, query, operator, value)],
                existing_results,
                column_info,
                limit,
                embedding_server_address,
            ),
            column_info,
        )

    elif isinstance(predicate, BoolExpr) and predicate.boolop == BoolExprType.AND_EXPR:
        field_query_list = []
        for a_pred in predicate.args:
            field_query_list.append(breakdown_unstructural_query(a_pred))

        return (
            retrieve_and_verify(
                node,
                field_query_list,
                existing_results,
                column_info,
                limit,
                embedding_server_address,
            ),
            column_info,
        )

    else:
        raise ValueError(
            "expects predicate to only contain atomic unstructural query, AND of them, or an NOT of atomic unsturctured query. However, this predicate is not: {}".format(
                RawStream()(predicate)
            )
        )


def execute_and(
    sql_dnf_predicates,
    node: SelectStmt,
    limit,
    cache: dict,
    fts_fields: List,
    embedding_server_address,
    select_username,
    select_userpswd,
):
    # there should not exist any OR expression inside sql_dnf_predicates

    if (
        isinstance(sql_dnf_predicates, BoolExpr)
        and sql_dnf_predicates.boolop == BoolExprType.AND_EXPR
    ):
        # find the structural part
        structural_predicates = tuple(
            filter(lambda x: if_all_structural(x), sql_dnf_predicates.args)
        )
        if len(structural_predicates) == 0:
            structural_predicates = None
        elif len(structural_predicates) == 1:
            structural_predicates = structural_predicates[0]
        else:
            structural_predicates = BoolExpr(
                boolop=BoolExprType.AND_EXPR, args=structural_predicates
            )

        # execute structural part
        structural_res, column_info = execute_structural_sql(
            node,
            structural_predicates,
            cache,
            fts_fields,
            select_username,
            select_userpswd,
        )

        free_text_predicates = tuple(
            filter(lambda x: not if_all_structural(x), sql_dnf_predicates.args)
        )
        if len(free_text_predicates) == 1:
            free_text_predicates = free_text_predicates[0]
        else:
            free_text_predicates = BoolExpr(
                boolop=BoolExprType.AND_EXPR, args=free_text_predicates
            )

        return execute_free_text_queries(
            node,
            free_text_predicates,
            structural_res,
            column_info,
            limit,
            embedding_server_address,
        )

    elif isinstance(sql_dnf_predicates, A_Expr) or (
        isinstance(sql_dnf_predicates, BoolExpr)
        and sql_dnf_predicates.boolop == BoolExprType.NOT_EXPR
    ):
        if if_all_structural(sql_dnf_predicates):
            return execute_structural_sql(
                node,
                sql_dnf_predicates,
                cache,
                fts_fields,
                select_username,
                select_userpswd,
            )
        else:
            all_results, column_info = execute_structural_sql(
                node, None, cache, fts_fields, select_username, select_userpswd
            )
            return execute_free_text_queries(
                node,
                sql_dnf_predicates,
                all_results,
                column_info,
                limit,
                embedding_server_address,
            )


def analyze_SelectStmt(
    node: SelectStmt,
    cache: dict,
    fts_fields: List,
    embedding_server_address: str,
    select_username: str,
    select_userpswd: str,
):
    limit = node.limitCount.val.ival if node.limitCount else -1
    sql_dnf_predicates = convert2dnf(node.whereClause)

    # if it is an OR, then order the predicates in structural -> unstructual
    # execute these predicates in order, until the limit is reached
    if (
        isinstance(sql_dnf_predicates, BoolExpr)
        and sql_dnf_predicates.boolop == BoolExprType.OR_EXPR
    ):
        choices = sorted(
            sql_dnf_predicates.args, key=lambda x: if_all_structural(x), reverse=True
        )
        res = []
        for choice in choices:
            choice_res, column_info = execute_and(
                choice,
                node,
                limit - len(res),
                cache,
                fts_fields,
                embedding_server_address,
                select_username,
                select_userpswd,
            )
            res.extend(choice_res)

            # at any time, if there is enough results, return that
            if len(res) >= limit and limit != -1:
                break

        return res, column_info

    elif (
        isinstance(sql_dnf_predicates, BoolExpr)
        and sql_dnf_predicates.boolop == BoolExprType.AND_EXPR
    ):
        return execute_and(
            sql_dnf_predicates,
            node,
            limit,
            cache,
            fts_fields,
            embedding_server_address,
            select_username,
            select_userpswd,
        )

    elif isinstance(sql_dnf_predicates, A_Expr) or (
        isinstance(sql_dnf_predicates, BoolExpr)
        and sql_dnf_predicates.boolop == BoolExprType.NOT_EXPR
    ):
        return execute_and(
            sql_dnf_predicates,
            node,
            limit,
            cache,
            fts_fields,
            embedding_server_address,
            select_username,
            select_userpswd,
        )
    else:
        raise ValueError(
            "Expects sql to be in DNF, but is not: {}".format(
                RawStream()(sql_dnf_predicates)
            )
        )


def suql_execute(
    suql,
    fts_fields=[],
    loggings="",
    disable_try_catch=False,
    embedding_server_address=EMBEDDING_SERVER_ADDRESS,
    select_username="select_user",
    select_userpswd="select_user",
    create_username="creator_role",
    create_userpswd="creator_role",
):
    """
    Main entry point to the SUQL Python-based compiler.

    Parameters:
    `suql` (str): The to-be-executed suql query,
    `fts_fields` (List[str], optional): Fields that should use PostgreSQL's Full Text Search (FTS) operators;
        The SUQL compiler would change certain string operators like "=" to use PostgreSQL's FTS operators.
        It uses `websearch_to_tsquery` and the `@@` operator to match against these fields.

    `loggings` (str, optional): Prefix for error case loggings. Errors are written to a "_suql_error_log.txt"
        file by default.
    `disable_try_catch` (bool, optional): whether to disable try-catch (errors would directly propagate to caller)
    `embedding_server_address` (str, optional): the embedding server address. Defaults to 'http://127.0.0.1:8501'
    `select_username` (str, optional): user name with select privilege in db. Defaults to "select_user".
    `select_userpswd` (str, optional): above user's password with select privilege in db. Defaults to "select_user".
    `create_username` (str, optional): user name with create privilege in db. Defaults to "creator_role".
    `create_userpswd` (str, optional): above user's password with create privilege in db. Defaults to "creator_role".

    Returns:
    `results` (List[[*]]): A list of returned database results. Each inner list stores a row of returned result
    `column_names` (List[str]): A list of database column names in the same order as `results`
    `cache` (Dict()): Debugging information from the SUQL compiler

    Example:
    ```
    suql_execute("SELECT * FROM restaurants WHERE
    answer(reviews, 'is this restaurant family-friendly?') = 'yes'", fts_fields=[("restaurants", "name")])
    ```

    In restaurants, one would likely need to apply FTS on the name of the restaurants
    since for queries that search by name, e.g.:
    ```
    suql_execute("SELECT * FROM restaurants WHERE name = 'mcdonalds'", fts_fields=[("restaurants", "name")])
    ```
    Ideally, this query should match against all `Mcdonald's`, as opposed to just 'mcdonalds'.
    FTS helps with such cases.
    """
    results, column_names, cache = suql_execute_single(
        suql,
        embedding_server_address,
        loggings,
        disable_try_catch,
        fts_fields,
        select_username,
        select_userpswd,
        create_username,
        create_userpswd,
    )
    if results == []:
        return results, column_names, cache
    all_no_results = True
    for result in results:
        for sub_result in result:
            if not (
                str(sub_result).lower() in "no information"
                or str(sub_result).lower() in "no info"
            ):
                all_no_results = False
                break
        if all_no_results == False:
            break

    # TODO: handle cases to go to multiple projection fields, from hybridQA
    return results, column_names, cache


def suql_execute_single(
    suql,
    embedding_server_address,
    loggings,
    disable_try_catch,
    fts_fields,
    select_username,
    select_userpswd,
    create_username,
    create_userpswd,
):
    results = []
    column_names = []
    cache = {}

    if disable_try_catch:
        visitor = SelectVisitor(
            fts_fields,
            embedding_server_address,
            select_username,
            select_userpswd,
            create_username,
            create_userpswd,
        )
        root = parse_sql(suql)
        visitor(root)
        second_sql = RawStream()(root)
        cache = visitor.serialize_cache()

        return execute_sql(second_sql, user=select_username, password=select_userpswd)
    else:
        try:
            visitor = SelectVisitor(
                fts_fields,
                embedding_server_address,
                select_username,
                select_userpswd,
                create_username,
                create_userpswd,
            )
            root = parse_sql(suql)
            visitor(root)
            second_sql = RawStream()(root)
            cache = visitor.serialize_cache()

            results, column_names, cache = execute_sql(
                second_sql, user=select_username, password=select_userpswd
            )
        except Exception as err:
            with open("_suql_error_log.txt", "a") as file:
                file.write(f"==============\n")
                file.write(f"{loggings}\n")
                file.write(f"{suql}\n")
                file.write(f"{str(err)}\n")
                traceback.print_exc(file=file)
        finally:
            visitor.drop_tmp_tables()
            return results, column_names, cache


if __name__ == "__main__":
    # print(suql_execute(sql, disable_try_catch=True, fts_fields=[("restaurants", "name")] )[0])
    with open("sql_free_text_support/test_cases.txt", "r") as fd:
        test_cases = fd.readlines()
    res = []
    for sql in test_cases:
        sql = sql.strip()
        i_res = suql_execute(sql, disable_try_catch=True)[0]
        res.append(i_res)
        with open("sql_free_text_support/test_cases_res.txt", "w") as fd:
            for i_res in res:
                fd.write(str(i_res))
                fd.write("\n")
