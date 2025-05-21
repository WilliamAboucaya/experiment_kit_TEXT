import pandas as pd

from utils.functions import get_neighbors, test_entailment, test_sentiment_analysis
import heapq
from itertools import combinations

# Function to convert lists to tuples, handling nested lists
def hashable_premise_serialized(premise_serialized):
    if isinstance(premise_serialized, list):
        return tuple(hashable_premise_serialized(i) for i in premise_serialized)
    return premise_serialized


def graph_explorator_bfs_optimized(df, goal, graph, model_nli_name, tokenizer_nli, model_nli,
                                   beam_width, max_depth, use_api, anchor_points_full_df, sentiment_task, ft):
    entailed_triples_df = pd.DataFrame(columns=["GOAL_TYPE", "SUBGOALS", "SUBGOALS_SERIALIZED", "SCORE", "NLI_LABEL"])
    priority_queue = []
    visited = set()  # Use a set to track visited premises for faster lookups

    # Buffer
    ##entailed_subsets_buffer = set()

    # Map from serialized premises to their precomputed similarity scores
    anchor_similarity_map = {
        hashable_premise_serialized(row["TRIPLE_SERIALIZED"]): row["SCORE"]
        for _, row in anchor_points_full_df.iterrows()
    }

    print("\nANCHOR SIMILARITY MAP:")
    print(anchor_similarity_map)

    # Store original beam width
    original_beam_width = beam_width

    # Initialize the priority queue with initial triples
    for _, row in df[
        ["GOAL_TYPE", "PREMISE", "HYPOTHESIS", "PREMISE_SERIALIZED", "ENTAILMENT", "NLI_LABEL"]].iterrows():

        if row['NLI_LABEL'] == "ENTAILMENT":
            entailed_triples_df.loc[len(entailed_triples_df)] = {
                "GOAL_TYPE": row['GOAL_TYPE'],
                "SUBGOALS": row["PREMISE"],
                "SUBGOALS_SERIALIZED": row["PREMISE_SERIALIZED"],
                "SCORE": row["ENTAILMENT"],
                "NLI_LABEL": row["NLI_LABEL"]
            }
        else:
            # Push a tuple with four elements: (negative entailment score, unique idx, row dict, depth)
            heapq.heappush(priority_queue,
                           (-row['ENTAILMENT'], _, row.to_dict(), 0))  # (negate score for max-heap, unique ID)

    # BFS with beam search, depth check, and score comparison
    while priority_queue:
        # Reset at the start of each new anchor triple
        current_beam_width = original_beam_width

        print('\nPRIORITY QUEUE:')
        print(priority_queue)

        # Pop a tuple with four elements
        current_entailment, idx, current_row, depth = heapq.heappop(priority_queue)

        print('\n-- CURRENT ENTAILMENT:')
        print(current_entailment)

        current_entailment = -current_entailment  # Convert back to positive entailment score

        print('\nCURRENT ENTAILMENT:')
        print(current_entailment)

        print('\nCURRENT ROW:')
        print(current_row)

        print('\nDEPTH:')
        print(depth)

        # If depth exceeds max_depth, stop further exploration
        if depth >= max_depth:
            continue

        # Convert 'PREMISE_SERIALIZED' to a fully hashable tuple
        premise_serialized_tuple = hashable_premise_serialized(current_row["PREMISE_SERIALIZED"])

        # Avoid revisiting the same triple
        if premise_serialized_tuple in visited:
            continue
        visited.add(premise_serialized_tuple)

        print('\nVISITED:')
        print(visited)

        # Get triple neighbors
        neighbor_triples = get_neighbors(current_row["PREMISE"], current_row["PREMISE_SERIALIZED"], graph)

        print("\nTRIPLE NEIGBORS:")
        print(neighbor_triples.to_string())

        if isinstance(neighbor_triples, pd.DataFrame) and neighbor_triples.empty:
            continue

        # Filter/Exclude triple neighbors already entailed and expand beam (+1)
        valid_neighbors = []
        beam_increase = 0

        for _, neighbor in neighbor_triples.iterrows():
            filtered_neighbor = neighbor["TRIPLE_NEIGHBOR"]

            if filtered_neighbor in ft:
                print(f"Skipping filtered triple: {filtered_neighbor}")
                continue

            neighbor_serialized = neighbor["TRIPLE_NEIGHBOR_SERIALIZED"]
            neighbor_hashed = hashable_premise_serialized(neighbor_serialized)

            already_entailed = False
            for e_serialized, label in zip(entailed_triples_df["SUBGOALS_SERIALIZED"],
                                           entailed_triples_df["NLI_LABEL"]):
                if label != "ENTAILMENT":
                    continue
                e_triples = e_serialized if isinstance(e_serialized[0], list) else [e_serialized]
                for triple in e_triples:
                    if hashable_premise_serialized([triple]) == neighbor_hashed:
                        already_entailed = True
                        break
                if already_entailed:
                    break

            if already_entailed:
                beam_increase += 1
                continue

            valid_neighbors.append(neighbor)

        current_beam_width += beam_increase
        print(
            f"\nAdjusted beam width: {current_beam_width} (original: {original_beam_width}, increase: {beam_increase})")

        if not valid_neighbors:
            continue

        valid_neighbors_df = pd.DataFrame(valid_neighbors)
        valid_neighbors_df["SIMILARITY_SCORE"] = valid_neighbors_df["TRIPLE_NEIGHBOR_SERIALIZED"].apply(
            lambda ps: anchor_similarity_map.get(hashable_premise_serialized(ps), 0)
        )
        valid_neighbors_df = valid_neighbors_df.nlargest(current_beam_width, "SIMILARITY_SCORE")

        print("\nVALID NEIGHBORS:")
        print(valid_neighbors_df.to_string())

        # --- Sentiment analysis on neighbor triples ---
        valid_neighbors_df["SENTIMENT"] = valid_neighbors_df["TRIPLE_NEIGHBOR_SERIALIZED"].apply(
            lambda triple_neighbors_serialized: test_sentiment_analysis(
                triple_neighbors_serialized[0],
                use_api,
                sentiment_task,
                neutral_predicates=["is a type of"]
            )[0]
        )

        print("\nSentiment analysis results on valid triple neighbors:")
        print(valid_neighbors_df.to_string())

        # Transform premises based on sentiment
        valid_neighbors_df["TRANSFORMED_PREMISE"] = valid_neighbors_df.apply(
            lambda row: "Prevent that " + row["TRIPLE_NEIGHBOR"]
            if row["SENTIMENT"] == "negative" else row["TRIPLE_NEIGHBOR"],
            axis=1
        )

        # Set GOAL_TYPE based on sentiment
        valid_neighbors_df["GOAL_TYPE"] = valid_neighbors_df["SENTIMENT"].apply(
            lambda s: "AVOID" if s == "negative" else "ACHIEVE"
        )

        # Concatenate with current anchor premise
        valid_neighbors_df["PREMISE_CONCATENATED"] = str(current_row["PREMISE"]) + ". " +  valid_neighbors_df["TRANSFORMED_PREMISE"]

        # Build final DataFrame
        concatenated_triples = pd.DataFrame({
            "GOAL_TYPE": valid_neighbors_df["GOAL_TYPE"],
            "PREMISE": valid_neighbors_df["PREMISE_CONCATENATED"],
            "HYPOTHESIS": goal,
            "PREMISE_SERIALIZED": valid_neighbors_df["TRIPLE_NEIGHBOR_SERIALIZED"].apply(
                lambda x: current_row["PREMISE_SERIALIZED"] + x
                if isinstance(current_row["PREMISE_SERIALIZED"], list)
                else [current_row["PREMISE_SERIALIZED"]] + x
            ),
            "SENTIMENT": valid_neighbors_df["SENTIMENT"]
        })

        print("\nTop beam_width triple neighbors:")
        print(concatenated_triples.to_string())

        # Apply entailment test to all triple neighbors and sort results by entailment score
        entailment_concatenate_triples_result = test_entailment(concatenated_triples, tokenizer_nli, model_nli_name, model_nli, use_api)
        entailment_concatenate_triples_result.sort_values(by="ENTAILMENT", ascending=False, inplace=True)

        print("\nENTAILMENT TEST RESULTS (CONCATENATED TRIPLES):")
        print(entailment_concatenate_triples_result.to_string())

        # The top `beam_width` neighbors
        top_k_neighbors = entailment_concatenate_triples_result

        # Track the highest entailment score and check for entailment
        found_entailment = False
        # Start with current entailment
        previous_entailment_score = current_entailment

        for _, neighbor_row in top_k_neighbors.iterrows():
            if neighbor_row["NLI_LABEL"] == "ENTAILMENT":
                triple_strs = [s.strip() for s in neighbor_row["PREMISE"].split(".") if s.strip()]
                print("\ntriples_strs:")
                print(triple_strs)

                last_triple = triple_strs[-1]
                print("\nlast triple:")
                print(last_triple)

                # Get the corresponding serialized triples
                serialized_triples = neighbor_row["PREMISE_SERIALIZED"]
                print("\nSerialized triples:")
                print(serialized_triples)

                # Create a mapping between string triples and their serialized versions
                triple_to_serialized = dict(zip(triple_strs, serialized_triples))

                all_subsets = [
                    list(combo) for i in range(1, len(triple_strs) + 1)
                    for combo in combinations(triple_strs, i)
                    if last_triple in combo
                ]

                print("\nALL SUBSETS (combination):")
                print(all_subsets)

                minimal_found = False

                for subset in sorted(all_subsets, key=len):
                    print("\nSubset:")
                    print(subset)

                    subset_premise = ". ".join(subset)
                    print("\nSubset string (already transformed):")
                    print(subset_premise)

                    if subset_premise in entailment_concatenate_triples_result["PREMISE"].values:
                        continue

                    # Get the corresponding serialized subset
                    subset_serialized = [triple_to_serialized[triple] for triple in subset]

                    subset_df = pd.DataFrame({
                        "GOAL_TYPE": [neighbor_row["GOAL_TYPE"]],
                        "PREMISE": [subset_premise],
                        "HYPOTHESIS": [goal],
                        "PREMISE_SERIALIZED": [subset_serialized]
                    })

                    subset_result = test_entailment(subset_df, tokenizer_nli, model_nli_name, model_nli, use_api)
                    print("\nSUBSET ENTAILMENT RESULT:")
                    print(subset_result.to_string())

                    if subset_result.iloc[0]["NLI_LABEL"] == "ENTAILMENT":
                        print("\nMinimal entailment found!")
                        entailed_triples_df = pd.concat([
                            entailed_triples_df,
                            pd.DataFrame({
                                "GOAL_TYPE": [neighbor_row["GOAL_TYPE"]],
                                "SUBGOALS": [subset_premise],
                                "SUBGOALS_SERIALIZED": [subset_serialized],
                                "SCORE": [subset_result.iloc[0]["ENTAILMENT"]],
                                "NLI_LABEL": ["ENTAILMENT"]
                            })
                        ])
                        found_entailment = True
                        minimal_found = True
                        print("\nSubset stored as entailing triple.")
                        break

                if not minimal_found:
                    print("\nNo minimal entailment found.")
                    entailed_triples_df = pd.concat([
                        entailed_triples_df,
                        pd.DataFrame({
                            "GOAL_TYPE": [neighbor_row["GOAL_TYPE"]],
                            "SUBGOALS": [neighbor_row["PREMISE"]],
                            "SUBGOALS_SERIALIZED": [neighbor_row["PREMISE_SERIALIZED"]],
                            "SCORE": [neighbor_row["ENTAILMENT"]],
                            "NLI_LABEL": ["ENTAILMENT"]
                        })
                    ])
                    found_entailment = True
                break
            elif neighbor_row["ENTAILMENT"] > previous_entailment_score:
                heapq.heappush(priority_queue, (-neighbor_row["ENTAILMENT"], _, neighbor_row.to_dict(), depth + 1))
                previous_entailment_score = neighbor_row["ENTAILMENT"]
            else:
                break

        if found_entailment:
            continue

    # Return sorted results by entailment score
    print("\n=> ENTAILED TRIPLES:")
    print(entailed_triples_df.sort_values(['SCORE'], ascending=[False]).to_string())
    entailed_triples_df.to_csv("entailed_triples.csv")
    return entailed_triples_df