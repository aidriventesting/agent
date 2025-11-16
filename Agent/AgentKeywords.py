from typing import Any, Dict, List, Optional

from Agent._step_runner import AgentStepRunner
from robot.api.deco import keyword


class AgentKeywords:
    """
    Robot Framework library exposing two high-level keywords:
    - Agent.Do <instruction>
    - Agent.VisualCheck <instruction>

    Agent.Do captures current UI context, composes a strict JSON prompt,
    calls the LLM with temperature=0, and executes the mapped AppiumLibrary action.
    
    Agent.VisualCheck captures a screenshot, sends it to AI for visual analysis,
    and provides detailed verification results with confidence scores.
    """

    ROBOT_LIBRARY_SCOPE = "GLOBAL"

    def __init__(self, llm_client: str = "openai", llm_model: str = "gpt-4o-mini"):
        self.engine = AgentStepRunner(llm_client=llm_client, llm_model=llm_model)

    # ----------------------- Public RF Keywords -----------------------
    def do(self, instruction: str):
        """Agent.Do <instruction>
        Example: Agent.Do    accepte les cookies
        """
        self.engine.do(instruction)

    def check(self, instruction: str):
        """Agent.VisualCheck <instruction>
        Example: Agent.VisualCheck    vérifier que l'écran affiche le logo de l'application
        """
        self.engine.visual_check(instruction)


    def autonumous(self, instruction: str):
        """Agent.Autonumous <instruction>
        This keyword is designed to autonomously plan and execute a test based on the 
        given single instruction.

        Example: Agent.Autonumous    Navigate to settings, change language to French, 
        ...    then go back to home screen and verify the interface is in French"""
        raise NotImplementedError("Agent.Autonumous is not implemented yet")