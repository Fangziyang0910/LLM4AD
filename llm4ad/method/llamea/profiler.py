from ...tools.profiler import ProfilerBase


class LLaMEAProfilerAdapter(ProfilerBase):
    """Profiler adapter for the external llamea package."""

    def log_solution_event(self, **payload):
        self.log_method_event(method="llamea", **payload)
