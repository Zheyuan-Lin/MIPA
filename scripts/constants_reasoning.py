###########################################################################################################
#                      Constants_reasoning.py : contain constants used for reasoning evaluation
###########################################################################################################
#



PROMPT_COT_REASONING = """
Analyze the discharge summary provided below to determine the presence of {phenotype}.
Use a detailed reasoning process as a medical doctor would to arrive at your conclusion. 
It is imperative that your response be accurate, as errors cannot be tolerated.

Discharge Summary:
{discharge_summary}

Provide your reasoning by following these exact steps:
1. Clinical Features:
   - List all relevant clinical features for {phenotype}
   - Indicate whether each feature supports or contradicts the diagnosis

2. Evidence Analysis:
   - Describe the strength of supporting evidence
   - Describe any contradicting evidence

3. Confounding Factors:
   - List any alternative explanations
   - Discuss potential confounders

4. Evidence Strength:
   - Rate overall evidence as: Strong, Moderate, or Weak
   - Explain your rating

5. Final Assessment:
   - Provide a concise summary of your medical reasoning
   - Explain how you arrived at your conclusion

Structure your response using these exact headings and bullet points.
"""


PROMPT_COT_OUTPUT = """
You have analyzed a discharge summary looking for evidence of {phenotype} through chain-of-thought reasoning.
Your task is to provide a final classification based on your previous analysis.

Previous Analysis:
{reasoning_analysis}

Output your conclusion in this exact JSON format:
{{
    "phenotype": "{phenotype}",
    "status": "1 or 0",
    "justification": "Your 50-token Justification Here",
    "confidence": "HIGH/MEDIUM/LOW"
}}
"""


PROMPT_SCOT_REASONING = """
Analyze the discharge summary provided below to determine the presence of {phenotype}.
As an experienced medical doctor, provide multiple independent reasoning paths to ensure consistency in your conclusion.

Discharge Summary:
{discharge_summary}

Perform three independent analyses:

1. Analysis Path 1 - Clinical Features:
- List all direct mentions and clinical manifestations of {phenotype}
- Evaluate the temporal relationship of these findings
- Provide initial conclusion based on clinical features (Present/Absent)

2. Analysis Path 2 - Treatment and Management:
- Identify medications, procedures, or interventions specific to {phenotype}
- Review follow-up plans and recommendations
- Provide conclusion based on management approach (Present/Absent)

3. Analysis Path 3 - Diagnostic Evidence:
- Review laboratory findings, imaging, or diagnostic tests
- Assess documented diagnoses and problem lists
- Provide conclusion based on diagnostic evidence (Present/Absent)

4. Consistency Check:
- Compare conclusions from all three paths
- Identify any discrepancies
- Resolve conflicts using clinical reasoning
- Determine final consensus

5. Final Assessment:
- Synthesize a final determination only if at least 2 out of 3 paths agree
- State the level of agreement between paths
- Provide final clinical reasoning for conclusion

Structure your response using these exact headings and bullet points.
"""

PROMPT_SCOT_OUTPUT = """
You have analyzed a discharge summary looking for evidence of {phenotype} through multiple analytical paths.
Your task is to provide a final classification based on your previous analysis.

Previous Analysis:
{reasoning_analysis}

Output your conclusion in this exact JSON format:
{{
    "phenotype": "{phenotype}",
    "status": "1 or 0",
    "justification": "Your 50-token Justification Here",
    "confidence": "HIGH if all paths agree, MEDIUM if 2/3 agree, LOW if significant discrepancy"
}}

DO NOT PROVIDE ANY OTHER SURROUNDING TEXT THAN THE JSON OUTPUT.
"""


PROMPT_TOT_REASONING = """
You are tasked with determining the presence of {phenotype} in the discharge summary provided below using a Tree-of-Thought approach.
This approach requires systematic exploration of multiple reasoning paths, with each path potentially branching into multiple sub-paths until reaching final decision leaves.

Discharge Summary:
{discharge_summary}

1. Key Findings (Group by relevance):
High Relevance:
- List findings that strongly support/refute {phenotype}
- Include specific lab values and clinical data

Medium Relevance:
- List findings that moderately relate to {phenotype}
- Include contextual clinical information

Low Relevance:
- List findings that might be relevant
- Include indirect clinical indicators

2. Branch Analysis (Explore these three core paths):

Common Branch Structure:
- Elements : findings in discharge summary relevant to the branch
- Evaluation : reasoning on the likelihood of the branch given the elements
- Initial Confidence: High/Medium/Low
- Decision: Generate relevant sub-branches
- For each sub-branch (x.1, x.2, etc.):
  - New evidence: [Additional findings/interpretation]
  - Impact on diagnosis
  - Confidence: High/Medium/Low
  - Decision: Conclude or generate deeper sub-branches
- For each final sub-branch (x.y.z):
  - New evidence: [Specific details]
  - Final Confidence: High/Medium/Low
  - Conclusion: Clear statement about presence/absence of {phenotype}

Branch 1: Evidence Supporting {phenotype}
- Elements:
  Supporting Evidence:
  - List direct mentions or clear diagnostic criteria
  - Include definitive lab values/clinical findings
  - Note temporal relationships
  Evaluation:
  - Assess strength of each piece of evidence
  - Consider reliability and specificity
[Follow Common Branch Structure]

Branch 2: Evidence Against {phenotype}
- Elements:
  Contradicting Evidence:
  - List findings that conflict with diagnosis
  - Include absent expected findings (missing critical criteria)
  - Document alternative explanations (other conditions explaining findings)
  Evaluation:
  - Assess impact on diagnostic certainty
  - Consider strength of contradictions
[Follow Common Branch Structure]

Branch 3: Alternative Diagnoses that could mislead us to think that {phenotype} is present.
- Elements:
  Alternative Explanations:
  - List other conditions explaining findings
  - Include overlapping syndromes
  - Document distinguishing features
  Evaluation:
  - Compare evidence support for each
  - Assess diagnostic criteria met
[Follow Common Branch Structure]

3. Backtracking Documentation:
- For each abandoned path:
  - Specify which branch/sub-branch
  - Provide explicit reasoning
  - Explain shift to alternative paths
  - Document evidence that prompted change

Instructions:

Remember that the objective is to think like a Physician and determine if the patient in the discharge summary has {phenotype}
Output Structure Requirements:
1. You MUST include all three main sections in order:
   - Key Findings (grouped by relevance)
   - Branch Analysis (all 3 branches with their conclusions)
   - Backtracking Documentation

2. Each Branch MUST:
   - Preserve hierarchical structure
   - Follow Common Branch Structure for all branches
   - Generate as many sub-branches as the evidence suggests (but contain at least two sub-branches)
   - Document reasoning at each branching point
   - Reach clear conclusions in final sub-branches
"""


PROMPT_TOT_OUTPUT = """
Based on the Tree-of-Thought analysis provided for determining the presence of {phenotype} in a discharge summary, synthesize the reasoning paths and provide a final determination.

Input Structure Being Analyzed:
1. Key Findings: Clinical evidence grouped by High/Medium/Low relevance
2. Branch Analysis: Three explored paths with sub-branches
   - Evidence Supporting {phenotype}
   - Evidence Against {phenotype}
   - Alternative Diagnoses
3. Backtracking Documentation: Reasoning for abandoned paths

Tree-of-Thought Analysis:
{reasoning_analysis}

Instructions:
Analyze the provided reasoning tree and:
1. Review all explored branches and their conclusions
2. Evaluate the strength of evidence in each path
3. Evaluate diagnostic specificity of key findings
4. Critique and recognize incomplete reasoning paths
5. Consider the backtracking decisions
6. Weigh contradicting evidence
7. Determine the most supported conclusion

Output your synthesis in this exact JSON format:
{{
    "phenotype": "{phenotype}",
    "status": "1 or 0",
    "justification": "50-token explanation emphasizing most definitive evidence",
    "confidence": "HIGH/MEDIUM/LOW",
    "selected_path": "Branch x: [Title] > Sub-branch x.y: [Title] (> Sub-branch x.y.z: [Title] if exists) | Final conclusion: [Conclusion]",
    "key_decision_points": [
        "List 4-6 critical points that drove the final decision, including key evidence, tipping points, and rejected alternatives"
    ]
}}
"""



PROMPT_GOT_REASONING_STEP_1 = """
You are tasked with determining the presence of {phenotype} in the discharge summary provided below using a reasoning graph approach. This approach requires systematic exploration of multiple reasoning paths to reach a final determination. The goal is to evaluate three key aspects:
1. Whether there is strong evidence supporting the presence of {phenotype}
2. Whether there is strong evidence against {phenotype}
3. Whether alternative diagnoses could be mimicking {phenotype}

The combination of these three paths will lead to a final determination about the presence of {phenotype} in this case.

Discharge Summary:
{discharge_summary}

1. Key Findings (Grouped by Relevance):

High Relevance:
- List findings that strongly support/refute {phenotype}
- Include specific lab values and clinical data

Medium Relevance:
- List findings that moderately relate to {phenotype}
- Include contextual clinical information

Low Relevance:
- List findings that might be relevant
- Include indirect clinical indicators

2. Reasoning Graph Construction:

The graph should be structured as a hierarchical representation with three main paths, each containing multiple subpaths. Each subpath should represent a distinct line of reasoning within the main path.

Node Types and Their Roles:

1. Finding Nodes:
   - Represent raw clinical data (labs, symptoms, exam findings)
   - Must include specific values/observations
   - Must specify if values are normal/abnormal

2. Evaluation Nodes:
   - Purpose: Synthesize multiple findings to form medical reasoning
   - Must include:
     a) Pattern Recognition: What clinical pattern do these findings suggest?
     b) Strength Assessment: How strongly do these findings support the pattern?
     c) Temporal Relationships: How do findings relate in time?
     d) Clinical Context: How do patient factors affect interpretation?
     e) Alternative Explanations: What else could explain these findings?

3. Conclusion Nodes:
   - Represent final interpretation for each subpath
   - Must clearly state support or rejection of the path's hypothesis
   - Must acknowledge uncertainty when present

Edge Types and Relationships:
- Establishes: Direct evidence establishing a fact/condition
- Strengthens: Additional evidence supporting a conclusion
- Suggests: Indirect evidence pointing toward a conclusion
- Contradicts: Evidence opposing a conclusion
- Leads_to: Temporal or causal relationship
- Modifies: Changes interpretation of another finding

Format for Nodes:
Node X (Type):Content - Confidence: Level
    Where:
    - X is a sequential number
    - Type is one of: Finding, Evaluation, Conclusion
    - Content follows type-specific requirements above
    - Level is High/Medium/Low

Format for Edges:
    Edge X: Source: Node Y, Target: Node Z, Relationship: Type
    Where:
    - X is a sequential number
    - Y is the source node number
    - Z is the target node number
    - Type is one of the edge types defined above

Construct the following three core paths:

Path 1: Evidence Supporting {phenotype}
- Create minimum two subpaths exploring different aspects of supporting evidence
- Each subpath should follow finding → evaluation → conclusion structure
- Conclusion nodes should specify strength of support for {phenotype}

Path 2: Evidence Against {phenotype}
- Create minimum two subpaths exploring different contradicting evidence
- Each subpath should follow finding → evaluation → conclusion structure
- Conclusion nodes should specify strength of evidence against {phenotype}

Path 3: Alternative Diagnoses that Could Mislead Us to Think that {phenotype} Is Present
- Create minimum two subpaths for different alternative diagnoses
- Each subpath should follow finding → evaluation → conclusion structure
- Conclusion nodes should assess likelihood of alternative diagnosis

Instructions:

1. Start with Key Findings grouped by relevance level

2. For each of the three main paths:
   - Create at least two distinct subpaths
   - Ensure each subpath has all three node types (Finding, Evaluation, Conclusion)
   - Make evaluation nodes comprehensive by addressing all required elements
   - Use appropriate edge types to show relationships
   - Maintain clear hierarchical structure through proper indentation

3. Ensure evaluation nodes explicitly address:
   - Pattern recognition in findings
   - Strength of evidence assessment
   - Temporal relationships
   - Clinical context
   - Alternative explanations

4. Document any abandoned reasoning paths separately

Example of a well-constructed subpath:

Path 1
 Node 1 (Finding): example - Confidence: High
 Node 4 (Evaluation): example - Confidence: Low
    Edge 1: Source: Node 1, Target: Node 4, Relationship: Establishes
    Edge 2: Source: Node 2, Target: Node 4, Relationship: Strengthens
Path 2
[…]
"""


PROMPT_GOT_REASONING_STEP_2 = """
The {reasoning_analysis} provided is from the analysis of a discharge summary, examining whether the patient presents with {phenotype}.
Apply Graph-of-Thought (GoT) methodology to determine presence/absence through:
- Aggregation of thoughts (combining nodes)
- Iterative refinement of conclusions 
- Interdependent reasoning across paths

Instructions:

1. Aggregation of all three paths:
  Using nodes/edges from previous
  Path 1 (Supporting Evidence)
  Path 2 (Opposing Evidence)
  Path 3 (Alternative Diagnoses))
  - Combine analysis and all three reasoning graphs into a single graph
  - Aggregate nodes (clinical findings)
  - Map relationships between findings
  - Determine {phenotype} presence/absence
  - Assign confidence based on documentation

2. Iterative Refinement:
  Systematically refine final reasoning graph
  - Re-evaluate findings after node aggregation
  - Update confidence based on documented evidence
  - Revise conclusions as clinical patterns emerge
  - Track confidence changes through iterations

3. Interdependent Reasoning:
  Analyze relationships in clinical presentation:
  - How documented symptoms in Path 1 influence Path 2
  - How alternative diagnoses explain patient's presentation
  - Cross-path interactions in clinical findings
  - Symptom overlap in discharge summary
  - Mutual reinforcement/contradiction in documentation

Required Output:

1. Final Reasoning Graph:
  Present refined discharge summary analysis showing:
  - Combined clinical findings with rationale
  - Connections between documented evidence
  - Strength of clinical correlations
  - Key interdependencies in presentation
  - Critical refinements in interpretation

Respect this format:

Edge Types and Relationships:
- Establishes: Direct evidence establishing a fact/condition
- Strengthens: Additional evidence supporting a conclusion
- Suggests: Indirect evidence pointing toward a conclusion
- Contradicts: Evidence opposing a conclusion
- Leads_to: Temporal or causal relationship
- Modifies: Changes interpretation of another finding

Final Graph
 Node 1 (Finding): example - Confidence: High
 Node 4 (Evaluation): example - Confidence: Low
    Edge 1: Source: Node 1, Target: Node 4, Relationship: Establishes
    Edge 2: Source: Node 2, Target: Node 4, Relationship: Strengthens
[...]

2. Final Determination:
  For {phenotype}:
  - Clear presence/absence based on discharge summary
  - Overall confidence level (HIGH/MEDIUM/LOW)
  - Key documented evidence clusters
  - Critical clinical patterns that influenced decision
  - Impact of alternative diagnoses consideration

Note: Focus solely on discharge summary evidence. No assumptions beyond documented findings. Maintain strict clinical reasoning based on available evidence.
"""

PROMPT_GOT_OUTPUT = """
Based on the Tree-of-Thought analysis provided for determining the presence of {phenotype}  in a discharge summary, synthesize the reasoning paths and provide a final determination.

Input Structure Being Analyzed:
1. Key Findings: Clinical evidence grouped by High/Medium/Low relevance
2. Branch Analysis: Three explored paths with sub-branches
   - Evidence Supporting {phenotype}
   - Evidence Against {phenotype}
   - Alternative Diagnoses
3. Backtracking Documentation: Reasoning for abandoned paths

Graph-of-Thought Analysis:
{reasoning_analysis}

Instructions:
Analyze the provided reasoning tree and:
1. Review all explored branches and their conclusions
2. Evaluate the strength of evidence in each path
3. Evaluate diagnostic specificity of key findings
4. Critique and recognize incomplete reasoning paths
5. Consider the backtracking decisions
6. Weigh contradicting evidence
7. Determine the most supported conclusion

Output your synthesis in this exact JSON format:
{{
    "phenotype": "{phenotype}",
    "status": "1 or 0",
    "justification": "50-token explanation emphasizing most definitive evidence",
    "confidence": "HIGH/MEDIUM/LOW",
    "selected_path": "Branch x: [Title] > Sub-branch x.y: [Title] (> Sub-branch x.y.z: [Title] if exists) | Final conclusion: [Conclusion]",
    "key_decision_points": [
        "List 4-6 critical points that drove the final decision, including key evidence, tipping points, and rejected alternatives"
    ]
}}
"""