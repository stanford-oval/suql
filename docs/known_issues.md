# Known issues

1. if you see error msgs similar to `PermissionError: [Errno 13] Permission denied: '/tmp/data-gym-cache/9b5ad71b2ce5302211f9c61530b329a4922fc6a4.2749b823-646b-45d7-9fcf-11414469d900.tmp'`. Refer to https://github.com/openai/tiktoken/issues/75. A likely solution is setting `TIKTOKEN_CACHE_DIR=""`.

2. A lot of times, Azure/OpenAI's chatGPT deployment's latency is unstable. We have experienced up to 10 minutes of latency for some inputs. These cases are rare (we estimate < 3% of cases), but they do happen from time to time. For those cases, if we cancel the request and re-issue them, then we typically can get a response in normal time. To counter this issue, we have implemented a max-wait-then-reissue functionality in our API calls. Under [this file](https://github.com/stanford-oval/genie-llm/blob/main/prompt_continuation.py), we have the following block:

```
if max_wait_time is None:
    max_wait_time = 0.005 * total_token + 1
```

This says that if a call to `llm_generate` does not set a `max_wait_time`, then it is dynamically calculated based on this linear function of `total_token`. This is imperfect, and we are erroring on the side of waiting longer (e.g., for an input with `1000` tokens, this would wait for 6 seconds, which might be too long). You can set a custom wait time, or disable this feature or together by setting `attempts = 0`.

3. The SUQL compiler right now uses the special character `^` when handling certain join-related optimizations. Please do not include this character `^` in your column names. (This restriction could be lifted in the future.)

4. When installing in python 3.10, encountered `ImportError: /home/oval/.conda/envs/.../lib/python3.10/site-packages/torch/lib/../../nvidia/cusparse/lib/libcusparse.so.12: undefined symbol: __nvJitLinkAddData_12_1, version libnvJitLink.so.12` (higher torch version could be from other packages in the env). Solved with `pip install torch==2.0.1`.
