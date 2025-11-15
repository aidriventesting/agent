from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict
from robot.api import logger


_LINE_PATTERN = re.compile(r"^\s*([a-zA-Z_]+)\s+(\d+):\s*(\{.*\})\s*$")


class Icon(TypedDict):
    type: str
    bbox: List[float]
    interactivity: bool
    content: str


class OmniParserResultProcessor:
    """
    Parse the raw OmniParser text payload into structured elements.

    Only `get_parsed_ui_elements` is intended to be used by consumers. 
    All helpers stay private to keep the parsing pipeline encapsulated.
    """

    def __init__(
        self,
        *,
        response_text: str,
        image_temp_path: Optional[str] = None,
    ) -> None:
        self._image_temp_path = image_temp_path or ""
        self._elements: List[OmniParserElement] = self._parse_response(response_text)
        logger.info(f"OmniParser detected {len(self._elements)} elements")
        if self._image_temp_path:
            logger.debug(f"Temporary image: {self._image_temp_path}")
        self._log_preview(limit=8)

    def get_parsed_ui_elements(self, *, element_type: Optional[str] = None) -> Dict[str, Icon]:
        """
        Return parsed elements keyed by their OmniParser label (whitespace removed).

        Parameters
        ----------
        element_type: Optional[str]
            - "interactive" for clickable items
            - "icon" or "text" for specific OmniParser element kinds
            - None (default) returns every element
        """
        if not element_type:
            filtered = self._elements
        else:
            element_type = element_type.strip().lower()
            if element_type == "interactive":
                filtered = [el for el in self._elements if el.interactivity]
            elif element_type == "all":
                filtered = self._elements
            else:
                filtered = [el for el in self._elements if el.element_type.lower() == element_type]

        return {self._element_key(element): element.to_icon() for element in filtered}

    @property
    def image_temp_path(self) -> str:
        """Return the temporary image path created by Gradio (optional, for debugging/display)."""
        return self._image_temp_path

    @staticmethod
    def _element_key(element: "OmniParserElement") -> str:
        """Build the dictionary key, ensuring the prefix (e.g. 'icon') stays untouched."""
        return element.label.replace(" ", "")

    def _parse_response(self, response_text: str) -> List[OmniParserElement]:
        elements: List[OmniParserElement] = []
        if not response_text:
            return elements

        for line in response_text.splitlines():
            clean_line = line.strip()
            if not clean_line:
                continue

            match = _LINE_PATTERN.match(clean_line)
            if not match:
                continue

            label_prefix, index_str, dict_payload = match.groups()
            attributes = self._safe_literal_eval(dict_payload)
            if attributes is None:
                continue

            # index_str is guaranteed by regex to be digits; direct cast keeps code simple
            index = int(index_str)
            element = self._build_element(label_prefix, index, attributes)
            if element:
                elements.append(element)

        return elements

    @staticmethod
    def _safe_literal_eval(payload: str) -> Optional[Dict[str, Any]]:
        try:
            parsed = ast.literal_eval(payload)
        except (SyntaxError, ValueError):
            return None
        if not isinstance(parsed, dict):
            return None
        return parsed

    def _build_element(
        self,
        label_prefix: str,
        index: int,
        attributes: Dict[str, Any],
    ) -> Optional[OmniParserElement]:
        element_type = str(attributes.get("type", "unknown"))
        raw_bbox = attributes.get("bbox", [])
        bbox = [float(v) for v in raw_bbox] if isinstance(raw_bbox, (list, tuple)) else []
        interactivity = bool(attributes.get("interactivity", False))
        content = str(attributes.get("content", ""))

        return OmniParserElement(
            index=index,
            label=f"{label_prefix} {index}",
            element_type=element_type,
            bbox=bbox,
            interactivity=interactivity,
            content=content,
        )

    def _log_preview(self, limit: int) -> None:
        if not self._elements:
            logger.debug("OmniParser parsed elements preview: none")
            return

        preview_lines = []
        for element in self._elements[:limit]:
            preview_lines.append(
                f"[{element.index}] {element.element_type} (interactive={element.interactivity}) "
                f"content='{element.content}' bbox={element.bbox}"
            )
        description = "\n".join(preview_lines)
        logger.debug(f"OmniParser parsed elements preview:\n{description}")



@dataclass(frozen=True)
class OmniParserElement:
    """
    Represents a single element detected by OmniParser.

    The OmniParser Hugging Face space returns textual lines containing Python-like
    dictionaries. We normalise them into a structured dataclass that is easier to
    consume downstream (e.g. by an LLM or by click coordinate utilities).
    """

    index: int
    label: str
    element_type: str
    bbox: List[float] = field(default_factory=list)
    interactivity: bool = False
    content: str = ""

    def to_icon(self) -> Icon:
        """Convert the element into the lightweight Icon payload."""
        return Icon(
            type=self.element_type,
            bbox=list(self.bbox),
            interactivity=self.interactivity,
            content=self.content,
        )


#quick test
if __name__ == "__main__":
    image_temp_path = "/private/var/folders/l1/gxvwfyt94jd6_8fzsgjp802r0000gp/T/gradio/f6f4505da872c2f7cb2149f8f664901903131ef3b98ee872134bdddef76d4da0/image.webp"
    
    response_text ="""
icon 0: {'type': 'text', 'bbox': [0.14722222089767456, 0.02478632517158985, 0.24074074625968933, 0.04059829190373421], 'interactivity': False, 'content': '22:37'}
icon 1: {'type': 'text', 'bbox': [0.19166666269302368, 0.11153846234083176, 0.7564814686775208, 0.1260683834552765], 'interactivity': False, 'content': 'Applications prevues pour vous'}
icon 2: {'type': 'text', 'bbox': [0.31203705072402954, 0.30811965465545654, 0.6898148059844971, 0.32606837153434753], 'interactivity': False, 'content': 'Toutes les applications.'}
icon 3: {'type': 'icon', 'bbox': [0.41938668489456177, 0.17028668522834778, 0.5745916366577148, 0.2660691440105438], 'interactivity': True, 'content': 'YouTube '}
icon 4: {'type': 'icon', 'bbox': [0.6046537756919861, 0.3621394634246826, 0.7612757086753845, 0.45402267575263977], 'interactivity': True, 'content': 'Astuces .... '}
icon 5: {'type': 'icon', 'bbox': [0.4223991930484772, 0.7786725759506226, 0.5733458995819092, 0.872641921043396], 'interactivity': True, 'content': 'Firefox '}
icon 6: {'type': 'icon', 'bbox': [0.4206276535987854, 0.36176764965057373, 0.5781549215316772, 0.4547795355319977], 'interactivity': True, 'content': 'Appium .... '}
icon 7: {'type': 'icon', 'bbox': [0.6055735945701599, 0.5012033581733704, 0.767842173576355, 0.5939653515815735], 'interactivity': True, 'content': 'Calculatri.... '}
icon 8: {'type': 'icon', 'bbox': [0.23331235349178314, 0.6399101614952087, 0.38924092054367065, 0.7330729365348816], 'interactivity': True, 'content': 'Contacts '}
icon 9: {'type': 'icon', 'bbox': [0.23282678425312042, 0.17132169008255005, 0.38811373710632324, 0.26554766297340393], 'interactivity': True, 'content': 'Gmail '}
icon 10: {'type': 'icon', 'bbox': [0.23351523280143738, 0.5011972188949585, 0.394944429397583, 0.5954245924949646], 'interactivity': True, 'content': 'Browser... '}
icon 11: {'type': 'icon', 'bbox': [0.41881510615348816, 0.5010577440261841, 0.5802904367446899, 0.5952942967414856], 'interactivity': True, 'content': 'Browser... '}
icon 12: {'type': 'icon', 'bbox': [0.6023470163345337, 0.170364111661911, 0.7598509192466736, 0.26560044288635254], 'interactivity': True, 'content': 'Photos '}
icon 13: {'type': 'icon', 'bbox': [0.22949214279651642, 0.36117684841156006, 0.39516758918762207, 0.4550669491291046], 'interactivity': True, 'content': 'Appareil ... '}
icon 14: {'type': 'icon', 'bbox': [0.6085647940635681, 0.6397818922996521, 0.7600852847099304, 0.7354426383972168], 'interactivity': True, 'content': 'Edge '}
icon 15: {'type': 'icon', 'bbox': [0.23595602810382843, 0.7781327962875366, 0.3900993764400482, 0.8724494576454163], 'interactivity': True, 'content': 'Files '}
icon 16: {'type': 'icon', 'bbox': [0.60603266954422, 0.7779232859611511, 0.7618451714515686, 0.8725081086158752], 'interactivity': True, 'content': 'Gmail '}
icon 17: {'type': 'icon', 'bbox': [0.7860805988311768, 0.500756025314331, 0.9516481757164001, 0.5934205651283264], 'interactivity': True, 'content': 'Can I See? '}
icon 18: {'type': 'icon', 'bbox': [0.7814667820930481, 0.1711302548646927, 0.945794403553009, 0.2654992640018463], 'interactivity': True, 'content': 'Horloge '}
icon 19: {'type': 'icon', 'bbox': [0.4229801297187805, 0.6397709250450134, 0.5726028084754944, 0.7346569895744324], 'interactivity': True, 'content': 'Drive '}
icon 20: {'type': 'icon', 'bbox': [0.052007611840963364, 0.501415491104126, 0.2086714506149292, 0.5951244831085205], 'interactivity': True, 'content': 'Browser... '}
icon 21: {'type': 'icon', 'bbox': [0.7850872874259949, 0.6392168402671814, 0.9538571834564209, 0.7333191633224487], 'interactivity': True, 'content': 'Enregistr... '}
icon 22: {'type': 'icon', 'bbox': [0.05158957466483116, 0.639639139175415, 0.2134605348110199, 0.7337194681167603], 'interactivity': True, 'content': 'Chrome '}
icon 23: {'type': 'icon', 'bbox': [0.7972462773323059, 0.7778764963150024, 0.9460236430168152, 0.8722485899925232], 'interactivity': True, 'content': 'Google '}
icon 24: {'type': 'icon', 'bbox': [0.05211879312992096, 0.7786644697189331, 0.2102147340774536, 0.8718963861465454], 'interactivity': True, 'content': 'Facebook '}
icon 25: {'type': 'icon', 'bbox': [0.04879764840006828, 0.3614121079444885, 0.20763833820819855, 0.455875962972641], 'interactivity': True, 'content': 'Agenda '}
icon 26: {'type': 'icon', 'bbox': [0.04571227356791496, 0.17190656065940857, 0.21723338961601257, 0.2657610774040222], 'interactivity': True, 'content': 'Facebook '}
icon 27: {'type': 'icon', 'bbox': [0.7702708840370178, 0.3528968095779419, 0.9393768310546875, 0.4539956748485565], 'interactivity': True, 'content': 'b.a.M '}
icon 28: {'type': 'icon', 'bbox': [0.4152655005455017, 0.9175739288330078, 0.5879263281822205, 1.0], 'interactivity': True, 'content': 'Messages'}
icon 29: {'type': 'icon', 'bbox': [0.6000481247901917, 0.917690098285675, 0.7701333165168762, 1.0], 'interactivity': True, 'content': 'Personalize'}
icon 30: {'type': 'icon', 'bbox': [0.785813570022583, 0.9168545603752136, 0.9510307312011719, 1.0], 'interactivity': True, 'content': 'Photos'}
icon 31: {'type': 'icon', 'bbox': [0.04773705452680588, 0.917568027973175, 0.21511192619800568, 1.0], 'interactivity': True, 'content': 'Handles'}
icon 32: {'type': 'icon', 'bbox': [0.23212599754333496, 0.9170367121696472, 0.3945298492908478, 1.0], 'interactivity': True, 'content': 'Google Maps'}
icon 33: {'type': 'icon', 'bbox': [0.24784258008003235, 0.02019941620528698, 0.3057074248790741, 0.046216681599617004], 'interactivity': True, 'content': 'Tibetan'}
icon 34: {'type': 'icon', 'bbox': [0.3633095324039459, 0.020187821239233017, 0.41940176486968994, 0.045822471380233765], 'interactivity': True, 'content': 'Headset'}
icon 35: {'type': 'icon', 'bbox': [0.30589643120765686, 0.019779615104198456, 0.3633996248245239, 0.04614794999361038], 'interactivity': True, 'content': 'Microsoft Edge'}
icon 36: {'type': 'icon', 'bbox': [0.42066264152526855, 0.020187685266137123, 0.47635725140571594, 0.04572726786136627], 'interactivity': True, 'content': 'Android'}
icon 37: {'type': 'icon', 'bbox': [0.9026451110839844, 0.01962120272219181, 0.9472880363464355, 0.04484112560749054], 'interactivity': True, 'content': 'PowerShell'}
icon 38: {'type': 'icon', 'bbox': [0.4819413423538208, 0.021491555497050285, 0.5270228981971741, 0.04548944905400276], 'interactivity': True, 'content': 'Dictation'}
icon 39: {'type': 'icon', 'bbox': [0.8514809012413025, 0.019348960369825363, 0.9021594524383545, 0.044579435139894485], 'interactivity': True, 'content': 'M0,0L9,5 4.5,5z'}
icon 40: {'type': 'icon', 'bbox': [0.7793554067611694, 0.10430154949426651, 0.836303174495697, 0.13246026635169983], 'interactivity': True, 'content': 'Close'}
icon 41: {'type': 'icon', 'bbox': [0.757146954536438, 0.019704006612300873, 0.8228548765182495, 0.044783253222703934], 'interactivity': True, 'content': 'Endnote'}
icon 42: {'type': 'icon', 'bbox': [0.9978095293045044, 0.14524270594120026, 1.0, 0.20872816443443298], 'interactivity': True, 'content': 'Stop'}
icon 43: {'type': 'icon', 'bbox': [0.8498612642288208, 0.08329697698354721, 0.9384395480155945, 0.13700133562088013], 'interactivity': True, 'content': 'More'}"""
    
    result = OmniParserResultProcessor(response_text=response_text, image_temp_path=image_temp_path)
    elements = result.get_parsed_ui_elements(element_type="interactive")
    
    # print("=" * 80)
    # print(f"len(elements): {len(elements)}")
    print("=" * 80)
    print(f"elements: {elements}")
    
