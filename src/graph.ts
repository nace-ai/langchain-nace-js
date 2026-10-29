import type { RunnableConfig } from "@langchain/core/runnables";

import { NaceClassifier, type NaceClassifierFields } from "./classifier.js";
import { NaceError } from "./errors.js";
import type { State } from "./state.js";
import type { Choice } from "./types.js";

const QUESTION_ID = "route";

export interface NaceConditionalEdgeOptions<GraphState> {
  /** The routing question. Its labels are the node names to route to. */
  question: Choice;
  /** Turns the graph state into Drex state, e.g. `(s) => s.messages`. */
  state: (graphState: GraphState) => State;
  /** Below this confidence the edge returns `lowConfidence`. */
  minConfidence: number;
  /** The node to go to when Drex is unsure. Defaults to `"human_review"`. */
  lowConfidence?: string;
  /** A configured classifier. Defaults to one built from `classifierOptions`. */
  classifier?: NaceClassifier;
  classifierOptions?: Omit<NaceClassifierFields, "questions">;
}

/**
 * A path function for LangGraph's `StateGraph.addConditionalEdges`.
 *
 * The edge asks Drex `question` about `state(graphState)` and returns the
 * chosen label. When Drex's confidence is below `minConfidence` it returns
 * `lowConfidence` instead, so the graph can go to a node that calls
 * `interrupt()` for a person.
 *
 * @example
 * ```ts
 * const edge = naceConditionalEdge({
 *   question: {
 *     type: "choice",
 *     instructions: "Who handles this?",
 *     criteria: { billing: null, technical: null },
 *   },
 *   state: (s) => s.messages,
 *   minConfidence: 0.7,
 * });
 * graph.addConditionalEdges("triage", edge, ["billing", "technical", "human_review"]);
 * ```
 */
export function naceConditionalEdge<GraphState>(
  options: NaceConditionalEdgeOptions<GraphState>,
): (graphState: GraphState, config?: RunnableConfig) => Promise<string> {
  const classifier = options.classifier ?? new NaceClassifier(options.classifierOptions);
  const lowConfidence = options.lowConfidence ?? "human_review";
  return async function naceConditionalEdge(graphState, config) {
    const response = await classifier.invoke(options.state(graphState), {
      ...config,
      runName: "NaceConditionalEdge",
      metadata: { ...config?.metadata, nace_min_confidence: options.minConfidence },
      questions: { [QUESTION_ID]: options.question },
    });
    const answer = response.choices[QUESTION_ID];
    if (answer === undefined) {
      throw new NaceError("Drex returned no answer for the routing question.");
    }
    return answer.confidence >= options.minConfidence ? answer.choice : lowConfidence;
  };
}
