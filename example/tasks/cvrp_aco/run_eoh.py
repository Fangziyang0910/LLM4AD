from llm4ad.method.eoh import EoH, EoHProfiler
from llm4ad.task.optimization.cvrp_aco import CVRPACOEvaluation
from llm4ad.tools.llm.vllm_openai_api import VLLMOpenAIAPI


def main() -> None:
    llm = VLLMOpenAIAPI(
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
        model="your-model",
        timeout=600,
    )
    method = EoH(
        llm=llm,
        evaluation=CVRPACOEvaluation(split="train"),
        profiler=EoHProfiler(log_dir="logs", log_style="complex"),
        max_sample_nums=1000,
        pop_size=20,
        num_samplers=4,
        num_evaluators=4,
    )
    method.run()


if __name__ == "__main__":
    main()
