"""Compatibility imports for the split BibTeX tool modules.

New code should import these tools from ``src.tools.builtin`` or their focused
implementation modules.
"""

from .citations.generation import GenerateBibtexTool, SummarizePaperTool, GenerateBibFromRefLibTool
from .citations.lookup import ScanCitationsTool, LookupPaperInfoTool, ListCiteKeysTool
from .citations.storage import WriteBibFileTool, AddReferenceTool
from .citations.validation import CompareCitationTool, ValidateAllCitationsTool

__all__ = [
    "GenerateBibtexTool", "SummarizePaperTool", "GenerateBibFromRefLibTool",
    "ScanCitationsTool", "LookupPaperInfoTool", "ListCiteKeysTool",
    "CompareCitationTool", "ValidateAllCitationsTool",
    "WriteBibFileTool", "AddReferenceTool",
]
