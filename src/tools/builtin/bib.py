"""Compatibility imports for the split BibTeX tool modules.

New code should import these tools from ``src.tools.builtin`` or their focused
implementation modules.
"""

from .citations.generation import GenerateBibtexTool, GenerateBibFromRefLibTool
from .citations.lookup import ScanCitationsTool, LookupPaperInfoTool, ListCiteKeysTool
from .citations.storage import AddReferenceTool
from .citations.validation import CompareCitationTool, ValidateAllCitationsTool

__all__ = [
    "GenerateBibtexTool", "GenerateBibFromRefLibTool",
    "ScanCitationsTool", "LookupPaperInfoTool", "ListCiteKeysTool",
    "CompareCitationTool", "ValidateAllCitationsTool",
    "AddReferenceTool",
]
