TAXONOMY_INIT_PROMPT = """
Role: You are a qualitative researcher using grounded theory to analyze employee recognition data.

<recognition_data>
{batch_content}
</recognition_data>

<task>
Create a high-level taxonomy of what employees are being recognized for.
- Focus on AREAS OF IMPACT, not just skills or behaviors.
- Include a concise description for each main category only.
- Do NOT include descriptions for subcategories.
- Return valid JSON matching the template below exactly.
</task>

<output_template>
{{
  "recognition_taxonomy": {{
    "CategoryName": {{
      "description": "One sentence describing this category.",
      "subcategories": ["Subcategory1", "Subcategory2", "Subcategory3"]
    }}
  }}
}}
</output_template>

Return only the JSON. No preamble, no explanation.
""".strip()


TAXONOMY_INCREMENTAL_PROMPT = """
Role: You are a qualitative researcher incrementally building a taxonomy using grounded theory.

<recognition_data>
{batch_content}
</recognition_data>

<existing_taxonomy>
{existing_taxonomy}
</existing_taxonomy>

<task>
1. Review the recognition data.
2. Identify ONLY new categories or subcategories NOT already in the existing taxonomy.
3. Return ONLY the new additions in the same JSON structure.
4. If nothing is new, return: {{"recognition_taxonomy": {{}}}}
5. Focus on AREAS OF IMPACT, not skills.
6. Include descriptions ONLY for new main categories.
</task>

Return only the JSON. No preamble, no explanation.
""".strip()


TAXONOMY_COMPRESS_PROMPT = """
Role: You are a qualitative researcher specializing in taxonomy optimization.

<current_taxonomy>
{taxonomy}
</current_taxonomy>

<task>
1. Identify categories or subcategories that are conceptually similar or redundant.
2. Merge overlapping items into concise, representative categories.
3. Maintain all important conceptual distinctions.
4. Return the optimized taxonomy in the same JSON structure.
5. Keep descriptions for main categories only.
</task>

Return only the JSON. No preamble, no explanation.
""".strip()


ANNOTATION_PROMPT = """
Role: You are an HR researcher classifying employee recognition messages.

<recognition_data>
{batch_content}
</recognition_data>

<taxonomy>
{taxonomy}
</taxonomy>

<task>
Classify each row in the recognition data.
Return a JSON ARRAY (list) with one object per row.
The array MUST start with [ and end with ]
Example of correct format for 3 rows:

[
  {{"row_id": "row1", "reason": "reason here", "category_name": "Name", "category_id": "A", "sub_category_name": "Name", "sub_category_id": "A1"}},
  {{"row_id": "row2", "reason": "reason here", "category_name": "Name", "category_id": "B", "sub_category_name": "Name", "sub_category_id": "B1"}},
  {{"row_id": "row3", "reason": "reason here", "category_name": "Name", "category_id": "C", "sub_category_name": "Name", "sub_category_id": "C1"}}
]

IMPORTANT:
- Start your response with [
- End your response with ]
- Include ALL rows, one entry per row
- Do NOT return a single object, return an ARRAY
</task>
""".strip()

HEALTH_CHECK_PROMPT = "Say OK"