"""内置工具 — 论文写作助手所有核心工具"""

from .pdf import ParsePDFTool, GetPaperTextTool, ParseAndStoreTool
from .search import SearchPapersTool, VerifyPaperTool
from .bib import (
    GenerateBibtexTool, SummarizePaperTool, ScanCitationsTool,
    LookupPaperInfoTool, CompareCitationTool, ValidateAllCitationsTool,
    WriteBibFileTool, AddReferenceTool, ListCiteKeysTool, GenerateBibFromRefLibTool,
)
from .tex import DeleteFileTool, ReadFileTool, WriteFileTool, ListFilesTool
from .template import ValidateTemplateTool
from .compile import CompileLatexTool, ParseLatexLogTool
from .dispatch import DispatchTaskTool
from .context import ReadContextTool
from .finish import FinishTool
from .library import ListPaperFilesTool, FindRelevantPapersTool, WriteLibraryTool

__all__ = [
    "ParsePDFTool", "GetPaperTextTool", "ParseAndStoreTool",
    "SearchPapersTool", "VerifyPaperTool",
    "GenerateBibtexTool", "SummarizePaperTool", "ScanCitationsTool",
    "LookupPaperInfoTool", "CompareCitationTool", "ValidateAllCitationsTool",
    "WriteBibFileTool", "AddReferenceTool",
    "ListCiteKeysTool", "GenerateBibFromRefLibTool",
    "ReadFileTool", "WriteFileTool", "ListFilesTool", "DeleteFileTool",
    "ValidateTemplateTool",
    "CompileLatexTool", "ParseLatexLogTool",
    "DispatchTaskTool",
    "ReadContextTool",
    "FinishTool",
    "ListPaperFilesTool", "FindRelevantPapersTool", "WriteLibraryTool",
]