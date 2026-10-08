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
      console.warn('Backend API returned non-OK response, falling back to mock data.');
      return DEFAULT_PIPELINE_RESULT;
    }

    const data = await response.json();
    return data as PipelineResult;
  } catch (err) {
    console.warn('Backend API unreachable, using local pipeline results.', err);
    return DEFAULT_PIPELINE_RESULT;
  }
}
