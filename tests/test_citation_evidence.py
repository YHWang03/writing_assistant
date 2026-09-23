import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.domain.paper import Paper
from src.domain.citation_evidence import classify, scan_claims, report_status, sources_for
from src.tools.builtin.citations.validation import ValidateAllCitationsTool, CompareCitationTool
from src.tools.builtin.tex import WriteFileTool

QUOTE = 'The proposed method converges on the tested model.'


class EvidenceTests(unittest.TestCase):
    def test_abstract_never_becomes_original_support(self):
        result = {'verdict':'✅', 'reason':'supported', 'source_index':0, 'evidence_quote':QUOTE}
        sources = [{'kind':'abstract', 'location':'abstract', 'text':QUOTE}]
        self.assertEqual(classify(result,sources,'claim')['status'],'abstract_support')
        sources[0]['kind'] = 'original'
        self.assertEqual(classify(result,sources,'claim')['status'],'insufficient')

    def test_fabricated_or_missing_quote_is_insufficient(self):
        sources = [{'kind':'original', 'location':'page=2', 'text':QUOTE}]
        for quote in ['Invented evidence not present', '', 'method']:
            result = {'verdict':'✅','reason':'ok','source_index':0,'evidence_quote':quote}
            self.assertEqual(classify(result,sources,'claim')['status'],'insufficient')
        self.assertEqual(classify(None,sources,'claim')['status'],'failed')

    def test_contradiction_needs_evidence(self):
        result = {'verdict':'❌','reason':'contradiction','source_index':0,'evidence_quote':QUOTE}
        sources = [{'kind':'abstract','location':'abstract','text':QUOTE}]
        self.assertEqual(classify(result,sources,'claim')['status'],'contradiction')
        self.assertEqual(classify(result,[],'claim')['status'],'insufficient')

    def test_repeated_citations_are_all_checked_and_comments_ignored(self):
        claims = scan_claims('A \\cite{x}\n% ignored \\cite{z}\nB \\citep{x,y}')
        self.assertEqual([c['cite_key'] for c in claims], ['x','x','y'])
        self.assertEqual(claims[-1]['line'],3)

    def test_old_library_cannot_claim_pdf_evidence(self):
        self.assertEqual(sources_for(Paper('a','Title','Author',2000), 'claim'),[])

    def test_direct_compare_returns_evidence_grade_not_raw_green_tick(self):
        tool = CompareCitationTool()
        with patch.object(tool,'_check_one',return_value={'verdict':'✅','reason':'title similar'}):
            result=json.loads(tool.execute('a','claim','title',QUOTE))
        self.assertEqual(result['status'],'insufficient')

    def test_reports_incremental_refresh_and_staleness(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            tex=root/'main.tex'
            tex.write_text('Claim A \\cite{a}\n' + 'x'*800 + '\nClaim B \\cite{b}',encoding='utf-8')
            bib=root/'references.bib'
            bib.write_text('@article{a,title={A}}\n@article{b,title={B}}',encoding='utf-8')
            refs=[Paper(k,k,'Author',2000,abstract=QUOTE) for k in ['a','b']]
            tool=ValidateAllCitationsTool()
            tool.set_reference_library(refs)
            def compare(citations):
                return ([{'cite_key':c['cite_key'],'verdict':'✅','reason':'supported',
                          'source_index':0,'evidence_quote':QUOTE} for c in citations],[])
            with patch('src.tools.builtin._safe_path._PROJECT_ROOT',root), patch.object(tool,'_batch_compare',side_effect=compare) as call:
                first=json.loads(tool.execute(str(tex)))
                self.assertEqual(first['counts']['abstract_support'],2)
                tool.execute(str(tex))
                self.assertEqual(call.call_args.args[0],[])
                bib.write_text('@article{a,title={Updated}}\n@article{b,title={B}}',encoding='utf-8')
                self.assertIn('已过期',report_status(tex,refs,mark_stale=True))
                self.assertTrue((root/'citation_report.txt').read_text(encoding='utf-8').startswith('[已过期]'))
                updated=json.loads(tool.execute(str(tex)))
                self.assertEqual(len(call.call_args.args[0]),1)
                self.assertEqual(sum(i['reused'] for i in updated['items']),1)
                self.assertIn('当前版本',report_status(tex,refs))
                denied=WriteFileTool().execute(file_path=str(root/'citation_report.txt'),content='all passed')
                self.assertTrue(denied.startswith('Error:'))

    def test_failure_not_counted_as_supported(self):
        with TemporaryDirectory() as directory:
            root=Path(directory); tex=root/'main.tex'
            tex.write_text('Claim \\cite{a}',encoding='utf-8')
            tool=ValidateAllCitationsTool()
            tool.set_reference_library([Paper('a','Title','Author',2000,abstract=QUOTE)])
            with patch('src.tools.builtin._safe_path._PROJECT_ROOT',root), patch.object(tool,'_batch_compare',return_value=([],[])):
                result=json.loads(tool.execute(str(tex)))
            self.assertEqual(result['counts']['failed'],1)
            self.assertNotIn('original_support',result['counts'])

    def test_pdf_is_never_opened_even_with_source_binding(self):
        ref = Paper('a','Title','Author',2000,abstract=QUOTE,source_pdf='paper.pdf',source_fingerprint='known')
        with patch.object(Path,'read_bytes',side_effect=AssertionError('PDF must not be read')):
            sources = sources_for(ref,'claim')
        self.assertEqual([s['kind'] for s in sources],['abstract'])

    def test_four_results_have_explicit_actions(self):
        sources=[{'kind':'abstract','location':'abstract','text':QUOTE}]
        for verdict, status in [('✅','abstract_support'),('❌','contradiction'),('❓','insufficient')]:
            result=classify({'verdict':verdict,'reason':'reason','source_index':0,'evidence_quote':QUOTE},sources,'claim')
            self.assertEqual(result['status'],status)
            self.assertEqual(result['passed'],status=='abstract_support')
            self.assertTrue(result['action'])
            if status=='contradiction': self.assertTrue(result['suggestion'])
        self.assertFalse(classify(None,sources,'claim')['passed'])


if __name__=='__main__': unittest.main()
