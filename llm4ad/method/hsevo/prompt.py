from __future__ import annotations

import copy

from ...base import Function


class HSEvoPrompt:
    USER_GENERATOR = """{seed} Your task is to write a {func_name} function for {problem_desc}
{func_desc}
"""

    CROSSOVER = """{user_generator}

### Better code
{func_signature_m1}
{code_method1}

### Worse code
{func_signature_m2}
{code_method2}

### Analyze & experience
- {analyze}
- {exp}

Your task is to write an improved function `{func_name}_v2` by COMBINING elements of two above heuristics base Analyze & experience.
Output the code within a Python code block: ```python ... ```, has comment and docstring (<50 words) to description key idea of heuristics design.

I'm going to tip $999K for a better heuristics! Let's think step by step."""

    MUTATION = """{user_generator}

Current heuristics:
{func_signature1}
{elitist_code}

Now, think outside the box write a mutated function `{func_name}_v2` better than current version.
You can use some hints below:
- {reflection}

Output code only and enclose your code with Python code block: ```python ... ```.
I'm going to tip $999K for a better solution!"""

    HARMONY_SEARCH = """[code]
{code_extract}

Now extract all threshold, weight or hardcode variable of the function make it become default parameters and give me a 'parameter_ranges' dictionary representation. Key of dict is name of variable. Value of key is a tuple in Python MUST include 2 float elements, first element is begin value, second element is end value corresponding with parameter.

- Output code only and enclose your code with Python code block: ```python ... ```.
- Output 'parameter_ranges' dictionary only and enclose your code with other Python code block: ```python ... ```."""

    FLASH_REFLECTION = """### List heuristics
Below is a list of design heuristics ranked from best to worst.
{lst_method}

### Guide
- Keep in mind, list of design heuristics ranked from best to worst. Meaning the first function in the list is the best and the last function in the list is the worst.
- The response in Markdown style and nothing else has the following structure:
"**Analysis:**
**Experience:**"
In there:
+ Meticulously analyze comments, docstrings and source code of several pairs (Better code - Worse code) in List heuristics to fill values for **Analysis:**.
Example: "Comparing (best) vs (worst), we see ...;  (second best) vs (second worst) ...; Comparing (1st) vs (2nd), we see ...; (3rd) vs (4th) ...; Comparing (second worst) vs (worst), we see ...; Overall:"

+ Self-reflect to extract useful experience for design better heuristics and fill to **Experience:** (<60 words).

I'm going to tip $999K for a better heuristics! Let's think step by step."""

    COMPREHENSIVE_REFLECTION = """Your task is to redefine 'Current self-reflection' paying attention to avoid all things in 'Ineffective self-reflection' in order to come up with ideas to design better heuristics.

### Current self-reflection
{curr_reflection}
{good_reflection}

### Ineffective self-reflection
{bad_reflection}

Response (<100 words) should have 4 bullet points: Keywords, Advice, Avoid, Explanation.
I'm going to tip $999K for a better heuristics! Let's think step by step."""

    SYSTEM_GENERATOR = """{seed} Your task is to design heuristics that can effectively solve optimization problems.
Your response outputs Python code and nothing else. Format your code as a Python code string: "```python ... ```"."""

    SYSTEM_REFLECTOR = "You are an expert in the domain of optimization heuristics. Your task is to provide useful advice based on analysis to design better heuristics."
    SYSTEM_HARMONY_SEARCH = "You are an expert in code review. Your task extract all threshold, weight or hardcode variable of the function make it become default parameters."

    SEED = """{seed_func}

Refer to the format of a trivial design above. Be very creative and give `{func_name}_v2`. Output code only and enclose your code with Python code block: ```python ... ```."""

    SCIENTISTS = [
        "You are an expert in the domain of optimization heuristics.",
        "You are Albert Einstein, relativity theory developer.",
        "You are Isaac Newton, the father of physics.",
        "You are Marie Curie, pioneer in radioactivity.",
        "You are Nikola Tesla, master of electricity.",
        "You are Galileo Galilei, champion of heliocentrism.",
        "You are Stephen Hawking, black hole theorist.",
        "You are Richard Feynman, quantum mechanics genius.",
        "You are Rosalind Franklin, DNA structure revealer.",
        "You are Ada Lovelace, computer programming pioneer.",
    ]

    def __init__(self, task_prompt: str, template_function: Function, external_knowledge: str = ""):
        self.task_prompt = task_prompt
        self.template_function = copy.deepcopy(template_function)
        self.func_name = template_function.name
        self.external_knowledge = external_knowledge or ""
        self.func_desc = self._build_func_desc()

    def _build_func_desc(self) -> str:
        docstring = self.template_function.docstring or ""
        docstring = docstring.replace('"""', '').strip()
        if docstring:
            return docstring
        return f"The {self.func_name} function must keep the given signature and solve this task: {self.task_prompt}"

    @staticmethod
    def compose(system: str, user: str) -> str:
        return f"{system.strip()}\n\n{user.strip()}"

    @classmethod
    def scientist(cls, index: int) -> str:
        return cls.SCIENTISTS[index % len(cls.SCIENTISTS)]

    def func_signature(self, version: int | str) -> str:
        ret = f" -> {self.template_function.return_type}" if self.template_function.return_type else ""
        return f"def {self.func_name}_v{version}({self.template_function.args}){ret}:"

    def _user_generator(self, seed: str) -> str:
        return self.USER_GENERATOR.format(
            seed=seed,
            func_name=self.func_name,
            problem_desc=self.task_prompt,
            func_desc=self.func_desc,
        )

    def _seed_prompt(self) -> str:
        seed_func = copy.deepcopy(self.template_function)
        seed_func.name = f"{self.func_name}_v1"
        return self.SEED.format(seed_func=str(seed_func), func_name=self.func_name)

    @staticmethod
    def _body(func: Function) -> str:
        return func.body.strip("\n")

    @staticmethod
    def _ranked(funcs: list[Function]) -> list[Function]:
        return sorted(funcs, key=lambda f: f.score, reverse=True)

    def initial_prompt(self, seed: str, long_term_reflection: str = "") -> str:
        system = self.SYSTEM_GENERATOR.format(seed=seed)
        user = "\n".join([
            self._user_generator(seed),
            self._seed_prompt(),
            long_term_reflection or self.external_knowledge,
        ])
        return self.compose(system, user)

    def flash_reflection_prompt(self, funcs: list[Function]) -> str:
        ranked = self._ranked(funcs)
        blocks = []
        for idx, func in enumerate(ranked):
            suffix = "th" if 11 <= idx + 1 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get((idx + 1) % 10, "th")
            blocks.append(f"[Heuristics {idx + 1}{suffix}]\n{str(func).strip()}\n")
        user = self.FLASH_REFLECTION.format(lst_method="\n".join(blocks))
        return self.compose(self.SYSTEM_REFLECTOR, user)

    def comprehensive_reflection_prompt(self, curr_reflection: str, good_reflection: str, bad_reflection: str) -> str:
        user = self.COMPREHENSIVE_REFLECTION.format(
            curr_reflection=curr_reflection or "None",
            good_reflection=good_reflection or "None",
            bad_reflection=bad_reflection or "None",
        )
        return self.compose(self.SYSTEM_REFLECTOR, user)

    def crossover_prompt(self, parents: list[Function], analysis: str, experience: str) -> str:
        ranked = self._ranked(parents)
        better, worse = ranked[0], ranked[1]
        seed = self.scientist(0)
        user = self.CROSSOVER.format(
            user_generator=self._user_generator(seed),
            func_signature_m1=self.func_signature(0),
            func_signature_m2=self.func_signature(1),
            code_method1=self._body(better),
            code_method2=self._body(worse),
            analyze=analysis or "None",
            exp=experience or "None",
            func_name=self.func_name,
        )
        return self.compose(self.SYSTEM_GENERATOR.format(seed=seed), user)

    def mutation_prompt(self, elite: Function, reflection: str) -> str:
        seed = self.scientist(0)
        user = self.MUTATION.format(
            user_generator=self._user_generator(seed),
            func_signature1=self.func_signature(1),
            elitist_code=self._body(elite),
            reflection=reflection or "None",
            func_name=self.func_name,
        )
        return self.compose(self.SYSTEM_GENERATOR.format(seed=seed), user)

    def harmony_search_prompt(self, func: Function) -> str:
        user = self.HARMONY_SEARCH.format(code_extract=str(func).strip())
        return self.compose(self.SYSTEM_HARMONY_SEARCH, user)

