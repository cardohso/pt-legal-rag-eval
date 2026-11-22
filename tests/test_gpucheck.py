from ptlegal_eval.gpucheck import evaluate_hardware, format_benchmark, main


def test_two_t4s_pass() -> None:
    ok, message = evaluate_hardware(["Tesla T4", "Tesla T4"])
    assert ok and "2" in message


def test_no_gpu_says_how_to_fix_it() -> None:
    ok, message = evaluate_hardware([])
    assert not ok and "GPU T4 x2" in message and "no gpu" in message.lower()


def test_one_gpu_or_the_wrong_kind_is_rejected_and_named() -> None:
    ok, message = evaluate_hardware(["Tesla P100-PCIE-16GB"])
    assert not ok and "P100" in message and "T4 x2" in message
    ok, message = evaluate_hardware(["Tesla T4"])
    assert not ok and "1" in message


def test_main_stops_with_status_1_on_wrong_hardware_and_runs_nothing_else() -> None:
    called = []
    out = []
    code = main([], names_fn=lambda: [], bench_fn=lambda: called.append(1) or {}, log=out.append)
    assert code == 1 and not called and any("GPU T4 x2" in line for line in out)


def test_main_runs_the_benchmark_on_the_right_hardware() -> None:
    bench = {"float16": {"tflops": 20.0, "finite": True}, "bfloat16": {"error": "RuntimeError: no"}}
    out = []
    code = main(
        [], names_fn=lambda: ["Tesla T4", "Tesla T4"], bench_fn=lambda: bench, log=out.append
    )
    assert code == 0 and any("float16" in line for line in out)


def test_main_reports_a_missing_torch_instead_of_crashing() -> None:
    def no_torch():
        raise ImportError("No module named torch")

    out = []
    assert main([], names_fn=no_torch, bench_fn=dict, log=out.append) == 1
    assert any("torch" in line for line in out)


def test_benchmark_formatting_shows_speed_errors_and_nonfinite_results() -> None:
    text = format_benchmark(
        {
            "float16": {"tflops": 18.5, "finite": True},
            "bfloat16": {"error": "RuntimeError: not supported"},
            "float32": {"tflops": 7.1, "finite": False},
        }
    )
    assert "18.5 TFLOPS" in text and "not supported" in text and "NOT finite" in text
