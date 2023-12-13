
from pglast.visitors import Visitor, Ancestor
import pglast
from pglast.ast import *
from pglast.enums.primnodes import BoolExprType, CoercionForm
from pglast.stream import RawStream
from pglast import parse_sql
from sympy import symbols, Symbol
from sympy.logic.boolalg import Or, And, Not, to_dnf
from pathlib import Path
import sys
# Append parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from postgresql_connection import execute_sql_with_column_info, execute_sql
from prompt_continuation import llm_generate
from utils import num_tokens_from_string
from copy import deepcopy
import time
import requests
import json
import random
import string
import concurrent.futures
from collections import defaultdict
from psycopg2 import Error as psyconpg2Error

SET_FREE_TEXT_FCNS = ["answer"]

def generate_random_string(length=12):
    characters = string.ascii_lowercase + string.digits
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string

class FreeTextFcnVisitor(Visitor):
    def __init__(self) -> None:
        super().__init__()
        self.set_free_text_fcns = SET_FREE_TEXT_FCNS
        self.res = False
    
    def __call__(self, node):
        super().__call__(node)
        
    def visit_FuncCall(self, ancestors, node : pglast.ast.FuncCall):
        for i in node.funcname:
            if i.sval in self.set_free_text_fcns:
                self.res = True
                return

def if_contains_free_text_fcn(node):
    visitor = FreeTextFcnVisitor()
    visitor(node)
    return visitor.res
         
class IfAllStructural(Visitor):
    def __init__(self) -> None:
        super().__init__()
        self.res = True
    
    def __call__(self, node):
        super().__call__(node)
        
    def visit_A_Expr(self, ancestors, node : A_Expr):
        if self.res is False:
            return
        
        def is_structural(expr):
            if isinstance(expr, FuncCall) and '.'.join(map(lambda x: x.sval, expr.funcname)) in SET_FREE_TEXT_FCNS:
                return False
            return True
        
        if not (is_structural(node.lexpr) and is_structural(node.rexpr)):
            self.res = False

class IfInvovlesSubquery(Visitor):
    def __init__(self) -> None:
        super().__init__()
        self.sublink = None
        
    def __call__(self, node):
        super().__call__(node)

    def visit_SubLink(self, ancestors, node : SubLink):
        print(ancestors)
        self.sublink = node.subselect

def if_all_structural(node):
    visitor = IfAllStructural()
    visitor(node)
    return visitor.res

class SelectVisitor(Visitor):
    def __init__(self) -> None:
        super().__init__()
        self.tmp_tables = []
        # this cache is to store classifier results for sturctured fields (e.g. cuisines in restaurants)
        self.cache = defaultdict(dict)
    
    def __call__(self, node):
        super().__call__(node)

    def visit_SelectStmt(self, ancestors, node : SelectStmt):
        if not node.whereClause:
            return

        # First, understand whether this involves subquery. If it does, then starts with that first and builds upwards
        # If that subquery in turn involves other subqueries, then recursive calls take care of it
        subquery_visitor = IfInvovlesSubquery()
        subquery_visitor(node)
        if subquery_visitor.sublink is not None:
            self.visit_SelectStmt(None, subquery_visitor.sublink)

        freeTextFcnVisitor = FreeTextFcnVisitor()
        freeTextFcnVisitor(node.whereClause)
        
        if freeTextFcnVisitor.res:
            tmp_table_name = "temp_table_{}".format(generate_random_string())
            self.tmp_tables.append(tmp_table_name)

            # main entry point for SUQL compiler optimization
            results, column_info = analyze_SelectStmt(node, self.cache)
            
            # based on results and column_info, insert a temporary table
            column_create_stmt = ",\n".join(list(map(lambda x: f'"{x[0]}" {x[1]}', column_info)))
            create_stmt = f"CREATE TABLE {tmp_table_name} (\n{column_create_stmt}\n); GRANT SELECT ON {tmp_table_name} TO select_user;"
            print("created table {}".format(tmp_table_name))
            execute_sql(create_stmt, user = "creator_role", password = "creator_role", commit_in_lieu_fetch=True, no_print=True)
            
            if results:
                # some special processing is needed for python dict types - they need to be converted to json
                json_indices = [index for index, element in enumerate(column_info) if element[1] in ('json', 'jsonb')]
                placeholder_str = ', '.join(['%s'] * len(results[0]))
                for result in results:
                    updated_results = tuple([json.dumps(element) if index in json_indices else element for index, element in enumerate(result)])
                    execute_sql(f"INSERT INTO {tmp_table_name} VALUES ({placeholder_str})", data = updated_results, user = "creator_role", password = "creator_role", commit_in_lieu_fetch=True)
            
            # finally, modify the existing sql with tmp_table_name
            node.fromClause = (RangeVar(relname=tmp_table_name, inh=True, relpersistence='p'),)
            node.whereClause = None
        else:
            classify_db_fields(node, self.cache)

    def serialize_cache(self):
        res = deepcopy(self.cache)
        for i in res:
            for j in res[i]:
                if isinstance(res[i][j], tuple):
                    res[i][j] = list(map(lambda x: x.sval, res[i][j]))
                else:
                    res[i][j] = [res[i][j].sval]
        return dict(res)

    def drop_tmp_tables(self):
        for tmp_table_name in self.tmp_tables:
            drop_stmt = f"DROP TABLE {tmp_table_name}"
            execute_sql(drop_stmt, user = "creator_role", password = "creator_role", commit_in_lieu_fetch=True)
    

class PredicateMapping():
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
    
    def predicate2symbol(_predicate : BoolExpr):
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
            return BoolExpr(boolop=BoolExprType.AND_EXPR, args=tuple(symbol2predicate(arg) for arg in symbol_predicate.args))
        if isinstance(symbol_predicate, Or):
            return BoolExpr(boolop=BoolExprType.OR_EXPR, args=tuple(symbol2predicate(arg) for arg in symbol_predicate.args))
        if isinstance(symbol_predicate, Not):
            return BoolExpr(boolop=BoolExprType.NOT_EXPR, args=tuple(symbol2predicate(arg) for arg in symbol_predicate.args))
        else:
            return predicate_mapping.retrieve_predicate(symbol_predicate)
        
    if (isinstance(predicate, A_Expr)):
        return predicate
    
    elif (isinstance(predicate, BoolExpr)):
        symbol_predicate = predicate2symbol(predicate)
        dnf_symbol_predicate = to_dnf(symbol_predicate)
        sql_expr = symbol2predicate(dnf_symbol_predicate)
        return sql_expr


def verify(document, field, query, operator, value):
    res = llm_generate(
        'prompts/verification.prompt',
        {
            "document": document,
            "field": field,
            "query": query,
            "operator": operator,
            "value": '"{}"'.format(value) if type(value) == str else value
        },
        engine='gpt-3.5-turbo-0613',
        temperature=0,
        stop_tokens=["\n"],
        max_tokens=30,
        postprocess=False)[0]
    
    if "the output is correct" in res.lower():
        return True
    else:
        return False

def verify_single_res(doc, field_query_list):
    # verify for each stmt, if any stmt fails to verify, exclude it
    all_found = True
    found_stmt = []
    for i, entry in enumerate(field_query_list):
        field, query, operator, value = entry
        if not verify(doc[i], field, query, operator, value):
            all_found = False
            break
        else:
            found_stmt.append("Verified answer({}, '{}') {} {} based on document: {}".format(field, query, operator, value, doc[i]))
    if all_found:
        print('\n'.join(found_stmt))
    elif found_stmt:
        print("partially verified: " + '\n'.join(found_stmt))
        
    
    return all_found

def parallel_filtering(fcn, source : list, limit, enforce_ordering = False):
    true_count = 0
    true_items = set()
    
    # build a dictionary with index -> true/false/None
    # which indicates whether an item has been verified
    ordered_results = {i : None for i in range(len(source))}
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(fcn, item): item for item in source}
        
        for future in concurrent.futures.as_completed(futures):
            item = futures[future]
            result = future.result()
            if result:
                true_count += 1
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
                res.append(item[0])
        return res

    return true_items

def retrieve_and_verify(node : SelectStmt, field_query_list, existing_results, column_info, limit, parallel = True):
    # field_query_list is a list of tuples, each entry of the tuple are:    
    # 0st: field: field to do retrieval on
    # 1st: query: query for retrieval model
    # 2nd: operator: operator to compare against
    # 3rd: value: value to compare against
    # 
    # existing_results: existing results to run retrieval on, this is a list of tuples
    # column_info: this is a list of tuples, first element is column name and second is type
    # limit: max number of returned results
    
    id_index = list(map(lambda x: x[0], column_info)).index('course_id') # TODO: this is hard-coded for restaurants. Automate fetching this -> I chnaged it from _id
    
    # get _id list from `existing_results`
    start_time = time.time()
    data = {
        "table_name": node.fromClause[0].relname, # TODO: implement support for JOIN statements
        "id_list": list(map(lambda x: x[id_index], existing_results)),
        "field_query_list": list(map(lambda x: (x[0], x[1]), field_query_list)),
        "top": limit * 20  # return 20 times the ordered amount, for GPT filtering purposes, TODO: this needs to be better planned
    }

    # Send a POST request
    response = requests.post('http://127.0.0.1:8509/search', json=data, headers={'Content-Type': 'application/json'})
    response.raise_for_status()
    parsed_result = response.json()["result"]
    
    filtered_parsed_result = []
    for res in parsed_result:
        if res not in filtered_parsed_result:
            filtered_parsed_result.append(res)
    parsed_result = filtered_parsed_result
    
    if parallel:
        # parallelize verification calls
        id_res = parallel_filtering(lambda x: verify_single_res(x[1], field_query_list), parsed_result, limit, enforce_ordering=True if node.sortClause is not None else False)
    else:
        id_res = []
        for each_res in parsed_result:
            if verify_single_res(each_res[1], field_query_list):
                id_res.append(each_res[0])
            
        
    end_time = time.time()
    print("retrieve + verification time {}s".format(end_time - start_time))
    
    return list(filter(lambda x: x[id_index] in id_res, existing_results))

def get_a_expr_field_value(node : A_Expr):
    if isinstance(node.lexpr, ColumnRef):
        column = node.lexpr
        value = node.rexpr
    elif isinstance(node.rexpr, ColumnRef) or isinstance(node.rexpr, ColumnRef):
        column = node.rexpr
        value = node.lexpr
        
    assert(isinstance(column, ColumnRef))
    assert(isinstance(value, A_Const))
    column_name = column.fields[0].sval
    value_res = value.val
    return column_name, value_res

def replace_a_expr_field(node : A_Expr, ancestors : Ancestor, new_value):
    def find_value_column(node : A_Expr):
        if isinstance(node.lexpr, ColumnRef):
            column = node.lexpr
            value = node.rexpr
        else:
            column = node.rexpr
            value = node.lexpr
            
        assert(isinstance(column, ColumnRef))
        assert(isinstance(value, A_Const))
        return column, value
    
    if (isinstance(new_value, tuple)):
        res = []
        for i in new_value:
            new_atomic_value = deepcopy(node)
            _, value = find_value_column(new_atomic_value)
            value.val = i
            res.append(new_atomic_value)
        new_node = BoolExpr(BoolExprType.OR_EXPR, args = tuple(res))
        if isinstance(ancestors.parent.node, BoolExpr):
            new_args = list(ancestors.parent.node.args)
            new_args[ancestors.member] = new_node
            ancestors.parent.node.args = tuple(new_args)
    else:
        _, value = find_value_column(node)
        value.val = new_value

def greedy_search_comma(input_string, predefined_list):
    chunks = input_string.split(',')
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

class StructuralClassification(Visitor):
    def __init__(self, node : SelectStmt, cache) -> None:
        super().__init__()
        self.node = node
        self.cache = cache
    
    def __call__(self, node):
        super().__call__(node)
        
    def visit_A_Expr(self, ancestors : Ancestor, node : A_Expr):
        assert(if_all_structural(node))        
        
        to_execute_node = deepcopy(self.node)
        
        # change projection to include everything
        to_execute_node.targetList = (ResTarget(val=ColumnRef(fields=(A_Star(), ))), )
        # reset all limits
        to_execute_node.limitCount = None
        # change predicates
        to_execute_node.whereClause = node
        # reset any groupby clause
        to_execute_node.groupClause = None
        
        try:
            res, column_infos = execute_sql_with_column_info(RawStream()(to_execute_node), unprotected=True)
        except psyconpg2Error:
            print("above error happens during ENUM classification attempts. Marking this predicate as returning answer.")
            res = True
            
        if not res:
            print("determined the above predicate returns no result")
            # try to classify into one of the known values
            # first, we need to find out what is the value here - some heuristics here to find out
            column_name, value_res = get_a_expr_field_value(node)
            
            # TODO handle none-text cases
            assert(isinstance(value_res, String))
            value_res_clear = value_res.sval
            
            print("determined column name: {}; value: {}".format(column_name, value_res_clear))
            
            # first check if this is already in the cache
            # TODO: for best performance move this before the execution above
            if column_name in self.cache and value_res_clear in self.cache[column_name]:
                replace_a_expr_field(node, ancestors, self.cache[column_name][value_res_clear])
            else:
                # first understand whether this field is of type TEXT or TEXT[]
                column_type = None
                for column_info in column_infos:
                    if column_info[0] == column_name:
                        column_type = column_info[1]
                        break
                
                assert(column_type is not None)
                if column_type.lower() == 'text[]':
                    # the SQL for getting results from TEXT[] is a bit complicated
                    # suppose the field is called cuisines, then this is the desired SQL
                    # SELECT DISTINCT unnest(cuisines) FROM restaurants;
                    to_execute_node.targetList = (ResTarget(val=FuncCall(
                        agg_distinct=False,
                        agg_star=False,
                        agg_within_group=False,
                        func_variadic=False,
                        funcformat=CoercionForm.COERCE_EXPLICIT_CALL,
                        args=(ColumnRef(fields=(column_name, ),),),
                        funcname=(String(sval=("unnest")),)
                    )), )
                    to_execute_node.distinctClause = (None, )
                else:
                    to_execute_node.targetList = (ResTarget(val=ColumnRef(fields=(column_name, ))), )
                    # TODO: maybe None would also work
                    to_execute_node.distinctClause = (ResTarget(val=ColumnRef(fields=(column_name, ))), )
                
                to_execute_node.sortClause = None
                to_execute_node.whereClause = None
                field_value_choices, _ = execute_sql_with_column_info(RawStream()(to_execute_node))
                # TODO deal with list problems?
                field_value_choices = list(map(lambda x:x[0], field_value_choices))
                field_value_choices.sort()
                
                # if a field has too many values (judged by tokens),
                # then it is not a enum field.
                # we give up on these (i.e., just use the existing predicate)
                if num_tokens_from_string('\n'.join(field_value_choices)) <= 3000:
                    res = llm_generate(
                        'prompts/field_classification.prompt',
                        {
                            "predicted_field_value": value_res_clear,
                            "field_value_choices": field_value_choices
                        },
                        engine='gpt-3.5-turbo-0613',
                        temperature=0,
                        stop_tokens=["\n"],
                        max_tokens=100,
                        postprocess=False)[0]
                    if res in field_value_choices:
                        replace_a_expr_field(node, ancestors, String(sval=(res)))
                        self.cache[column_name][value_res_clear] = String(sval=(res))
                    # tries to parse it as a list
                    else:
                        parsed_res = greedy_search_comma(res, field_value_choices)
                        if parsed_res:
                            parsed_res = tuple(map(lambda x: String(sval=(x)), parsed_res))
                            replace_a_expr_field(node, ancestors, parsed_res)
                            self.cache[column_name][value_res_clear] = parsed_res


def classify_db_fields(node : SelectStmt, cache):
    # we expect all atomic predicates under `predicate` to only involve stru fields
    # (no `answer` function)
    # the goal of this function is to determine which predicate leads to no results
    # for a field without results, try to classify into one of the existing fields
    visitor = StructuralClassification(node, cache)
    visitor(node)


def execute_structural_sql(node : SelectStmt, predicate : BoolExpr, cache : dict):
    node = deepcopy(node)
    # change projection to include everything
    node.targetList = (ResTarget(val=ColumnRef(fields=(A_Star(), ))), )
    # reset all limits
    node.limitCount = None
    node.limitOffset = None
    # change predicates
    node.whereClause = predicate
    
    # only queries that involve only structural parts can be executed
    assert(if_all_structural(node))
    
    # deal with sturctural field classification
    classify_db_fields(node, cache)
    
    sql = RawStream()(node)
    print("execute_structural_sql executing sql: {}".format(sql))
    return execute_sql_with_column_info(sql)
    

def execute_free_text_queries(node, predicate : BoolExpr, existing_results, column_info, limit):
    # the predicate should only contain an atomic unstructural query
    # or an AND of multiple unstructural query (NOT of an unstructural query is considered to be atmoic)
    def breakdown_unstructural_query(predicate : A_Expr):
        assert(if_contains_free_text_fcn(predicate.lexpr) or if_contains_free_text_fcn(predicate.rexpr))
        if if_contains_free_text_fcn(predicate.lexpr) and if_contains_free_text_fcn(predicate.rexpr):
            raise ValueError("cannot optimize for predicate containing free text functions on both side of expression: {}".format(RawStream()(predicate)))
        
        free_text_clause = predicate.lexpr if if_contains_free_text_fcn(predicate.lexpr) else predicate.rexpr
        value_clause = predicate.lexpr if not if_contains_free_text_fcn(predicate.lexpr) else predicate.rexpr
        assert(isinstance(free_text_clause, FuncCall))
        assert(isinstance(value_clause, A_Const))
        
        query_lst = list(filter(lambda x: isinstance(x, A_Const), free_text_clause.args))
        assert(len(query_lst) == 1)
        query = query_lst[0].val.sval
        
        field_lst = list(filter(lambda x: isinstance(x, ColumnRef), free_text_clause.args))
        assert(len(field_lst) == 1)
        field = field_lst[0].fields[0].sval
        
        operator = predicate.name[0].sval
        if isinstance(value_clause.val, String):
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
        return retrieve_and_verify(node, [(field, query, operator, value)], existing_results, column_info, limit), column_info

    elif isinstance(predicate, BoolExpr) and predicate.boolop == BoolExprType.AND_EXPR:
        field_query_list = []
        for a_pred in predicate.args:
            field_query_list.append(breakdown_unstructural_query(a_pred))
        
        return retrieve_and_verify(node, field_query_list, existing_results, column_info, limit), column_info
    
    else:
        raise ValueError("expects predicate to only contain automatic unstructural query or AND of them, but predicate is not: {}".format(RawStream()(predicate)))

def execute_and(sql_dnf_predicates, node : SelectStmt, limit, cache : dict):
    # there should not exist any OR expression inside sql_dnf_predicates
    
    if isinstance(sql_dnf_predicates, BoolExpr) and sql_dnf_predicates.boolop == BoolExprType.AND_EXPR:
        # find the structural part
        structural_predicates = tuple(filter(lambda x: if_all_structural(x), sql_dnf_predicates.args))
        if len(structural_predicates) == 0:
            structural_predicates =  None
        elif len(structural_predicates) == 1:
            structural_predicates =  structural_predicates[0]
        else:
            structural_predicates = BoolExpr(boolop=BoolExprType.AND_EXPR, args = structural_predicates)
        
        # execute structural part
        structural_res, column_info = execute_structural_sql(node, structural_predicates, cache)
        
        free_text_predicates = tuple(filter(lambda x: not if_all_structural(x), sql_dnf_predicates.args))
        if len(free_text_predicates) == 1:
            free_text_predicates = free_text_predicates[0]
        else:
            free_text_predicates = BoolExpr(boolop=BoolExprType.AND_EXPR, args = free_text_predicates)
        
        return execute_free_text_queries(node, free_text_predicates, structural_res , column_info, limit)
    
    elif isinstance(sql_dnf_predicates, A_Expr) or (isinstance(sql_dnf_predicates, BoolExpr) and sql_dnf_predicates.boolop == BoolExprType.NOT_EXPR):
        if if_all_structural(sql_dnf_predicates):
            return execute_structural_sql(node, sql_dnf_predicates, cache)
        else:
            all_results, column_info = execute_structural_sql(node, None, cache)
            return execute_free_text_queries(node, sql_dnf_predicates, all_results, column_info, limit)


def analyze_SelectStmt(node : SelectStmt, cache : dict):
    limit = node.limitCount.val.ival if node.limitCount else -1
    sql_dnf_predicates = convert2dnf(node.whereClause)

    # if it is an OR, then order the predicates in structural -> unstructual
    # execute these predicates in order, until the limit is reached
    if isinstance(sql_dnf_predicates, BoolExpr) and sql_dnf_predicates.boolop == BoolExprType.OR_EXPR:
        choices = sorted(sql_dnf_predicates.args, key = lambda x: if_all_structural(x), reverse=True)
        res = []
        for choice in choices:
            choice_res, column_info = execute_and(choice, node, limit - len(res), cache)
            res.extend(choice_res)
            
            # at any time, if there is enough results, return that 
            if len(res) >= limit and limit != -1:
                break
        
        return res, column_info
    
    elif isinstance(sql_dnf_predicates, BoolExpr) and sql_dnf_predicates.boolop == BoolExprType.AND_EXPR:
        return execute_and(sql_dnf_predicates, node, limit, cache)
    
    elif isinstance(sql_dnf_predicates, A_Expr) or (isinstance(sql_dnf_predicates, BoolExpr) and sql_dnf_predicates.boolop == BoolExprType.NOT_EXPR):
        return execute_and(sql_dnf_predicates, node, limit, cache)
        
    else:
        raise ValueError("Expects sql to be in DNF, but is not: {}".format(RawStream()(sql_dnf_predicates)))
            

if __name__ == "__main__":
    # root = parse_sql("SELECT *, summary(reviews) FROM restaurants WHERE 'brunch' = ANY (cuisines) AND location = 'San Francisco' ORDER BY num_reviews DESC LIMIT 3;")
    # root = parse_sql("SELECT *, summary(reviews) FROM restaurants WHERE answer(reviews, 'is this a pet-friendly restaurant') = 'Yes' LIMIT 4")
    # root = parse_sql("SELECT *, summary(reviews) FROM restaurants WHERE location = 'Sunnyvale' AND answer(reviews, 'does this restaurant have live music?') = 'Yes' LIMIT 4")
    # root = parse_sql("SELECT *, summary(reviews) FROM restaurants WHERE answer(reviews, 'what is the price range') <= 20 LIMIT 4")
    # root = parse_sql("SELECT *, summary(reviews) FROM restaurants WHERE location = 'Sunnyvale' AND answer(reviews, 'does this restaurant have live music?') = 'Yes' AND answer(reviews, 'does this restaurant have good ambiance') = 'Yes' LIMIT 4")
    # root = parse_sql("SELECT *, summary(reviews), answer(reviews, 'is this restaurant family-friendly?'), answer(reviews, 'what is the atmosphere?') FROM restaurants WHERE answer(reviews, 'do you find this restaurant to be family-friendly?') = 'Yes' AND answer(reviews, 'what is the atmosphere?') = 'Good' LIMIT 1;")
    # root = parse_sql("SELECT *, summary(reviews) FROM restaurants WHERE answer(reviews, 'does this restaurant have outdoor seating?') = 'Yes' OR answer(reviews, 'does this restaurant have a garden') = 'Yes' AND location = 'Palo Alto' LIMIT 1;")
    # root = parse_sql("SELECT *, summary(reviews) FROM restaurants WHERE (answer(reviews, 'does this restaurant have outdoor seating?') = 'Yes' OR answer(reviews, 'does this restaurant have a garden') = 'Yes') AND location = 'Palo Alto' LIMIT 1;")
    # root = parse_sql("SELECT *, summary(reviews) FROM restaurants WHERE answer(popular_dishes, 'does it contain grilled cheese?') = 'Yes' AND answer(reviews, 'is this restaurant family-friendly') = 'Yes' LIMIT 1;")
    # root = parse_sql("SELECT * FROM restaurants WHERE 'cafe' = ANY (cuisines) LIMIT 1;")
    # root = parse_sql("SELECT *, summary(reviews) FROM restaurants WHERE 'japanese' = ANY (cuisines) AND location = 'downtown SF' ORDER BY rating DESC, num_reviews DESC LIMIT 1;")
    # root = parse_sql("SELECT *, summary(reviews), answer(reviews, 'is this restaurant family-friendly?') FROM restaurants WHERE 'chinese' = ANY (cuisines) AND location = 'San Francisco' AND answer(reviews, 'do you find this restaurant to be family-friendly?') = 'Yes' LIMIT 1;")
    # root = parse_sql("SELECT *, summary(reviews), answer(reviews, 'is this restaurant authentic?') FROM restaurants WHERE 'mexican' = ANY (cuisines) AND answer(reviews, 'is this restaurant family-friendly?') = 'Yes' AND rating >= 4.0 ORDER BY rating DESC LIMIT 3;")
    root = parse_sql(" SELECT title, summary(description) FROM courses WHERE 'WAY-ER' = ANY(general_requirements) LIMIT 3;") # FIXIT
    visitor = SelectVisitor()
    visitor(root)
    print(RawStream()(root))
    print(visitor.serialize_cache())

    # results
    print("Executed results")
    results, column_names, _ = execute_sql(RawStream()(root))
    print(results)
    
    visitor.drop_tmp_tables()