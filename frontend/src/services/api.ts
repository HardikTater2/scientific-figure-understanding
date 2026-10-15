import type { PipelineResult } from '../types';
import { DEFAULT_PIPELINE_RESULT } from '../data/mockData';

export async function runPipeline(
  paperPath: string = 'sample.pdf',
  figurePath: string = 'figure.png',
  query: string = 'What does this figure show?'
): Promise<PipelineResult> {
  try {
    const response = await fetch('/api/run-pipeline', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        paper_path: paperPath,
        figure_path: figurePath,
        query: query
      })
    });

    if (!response.ok) {
      console.warn('Backend API returned non-OK response; showing built-in demo data (flagged as mock).');
      return DEFAULT_PIPELINE_RESULT;
    }

    const data = await response.json();
    return data as PipelineResult;
  } catch (err) {
    console.warn('Backend API unreachable; showing built-in demo data (flagged as mock).', err);
    return DEFAULT_PIPELINE_RESULT;
  }
}

export async function checkBackendHealth(): Promise<boolean> {
  try {
    const response = await fetch('/api/health');
    return response.ok;
  } catch {
    return false;
  }
}

export async function uploadAndRunPipeline(
  figureFile: File,
  paperFile: File | null,
  query: string
): Promise<PipelineResult> {
  const formData = new FormData();
  formData.append('figure_file', figureFile);
  if (paperFile) {
    formData.append('paper_file', paperFile);
  }
  formData.append('query', query);

  const response = await fetch('/api/upload-and-run', {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`Pipeline run failed (${response.status}): ${errText}`);
  }

  const data = await response.json();
  data.is_mock_data = false;
  return data as PipelineResult;
}

