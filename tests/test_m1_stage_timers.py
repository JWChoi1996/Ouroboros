import json
import unittest


class FakeClock:
    def __init__(self, *values):
        self._values = iter(values)

    def __call__(self):
        return next(self._values)


class FakeEvent:
    def __init__(self, elapsed_ms):
        self.elapsed_ms = elapsed_ms
        self.recorded = False

    def record(self):
        self.recorded = True

    def elapsed_time(self, other):
        return other.elapsed_ms


class FakeCudaBackend:
    def __init__(self, *elapsed_ms):
        values = []
        for elapsed in elapsed_ms:
            values.extend((0.0, elapsed))
        self._events = iter(FakeEvent(value) for value in values)
        self.synchronize_calls = 0

    def event(self):
        return next(self._events)

    def synchronize(self):
        self.synchronize_calls += 1


class M1StageTimerTests(unittest.TestCase):
    def test_cpu_stages_accumulate_serializable_milliseconds(self):
        from on_device.motivation_m1.stage_timers import StageTimer

        timer = StageTimer(enabled=True, clock=FakeClock(1.0, 1.025, 2.0, 2.010))

        with timer.cpu("model_preprocess"):
            pass
        with timer.cpu("model_preprocess"):
            pass

        totals = timer.totals()
        self.assertAlmostEqual(totals["model_preprocess"], 35.0)
        self.assertEqual(json.loads(json.dumps(totals)), totals)

    def test_disabled_timer_callback_is_a_reusable_no_op(self):
        from on_device.motivation_m1.stage_timers import StageTimer

        timer = StageTimer()
        stage = timer.callback("attention")

        with stage():
            pass
        with stage():
            pass

        self.assertEqual(timer.totals(), {})

    def test_cuda_events_are_resolved_after_one_explicit_synchronization(self):
        from on_device.motivation_m1.stage_timers import StageTimer

        backend = FakeCudaBackend(2.5, 3.5)
        timer = StageTimer(enabled=True, cuda_backend=backend)

        with timer.cuda("attention"):
            pass
        with timer.cuda("attention"):
            pass

        self.assertEqual(backend.synchronize_calls, 0)
        timer.synchronize()
        self.assertEqual(backend.synchronize_calls, 1)
        self.assertEqual(timer.totals()["attention"], 6.0)
        self.assertEqual(backend.synchronize_calls, 1)

    def test_model_forward_is_total_minus_preprocess(self):
        from on_device.motivation_m1.stage_timers import StageTimer

        timer = StageTimer(
            enabled=True, cuda_backend=FakeCudaBackend(12.0, 4.5)
        )

        with timer.cuda("model_total"):
            with timer.cuda("model_preprocess"):
                pass

        totals = timer.totals()
        self.assertEqual(totals["model_total"], 12.0)
        self.assertEqual(totals["model_preprocess"], 4.5)
        self.assertEqual(totals["model_forward"], 7.5)

    def test_derived_model_forward_is_never_negative(self):
        from on_device.motivation_m1.stage_timers import StageTimer

        timer = StageTimer(
            enabled=True, cuda_backend=FakeCudaBackend(3.0, 4.0)
        )

        with timer.cuda("model_total"):
            with timer.cuda("model_preprocess"):
                pass

        self.assertEqual(timer.totals()["model_forward"], 0.0)


if __name__ == "__main__":
    unittest.main()
