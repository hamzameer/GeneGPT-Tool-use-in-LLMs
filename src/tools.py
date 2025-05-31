"""This module contains the tools definitions for the LLM."""

import json
import os
import re
import time

import requests
from dotenv import load_dotenv

load_dotenv()

NCBI_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
BLAST_BASE_URL = "https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi"
NCBI_API_KEY = os.getenv("NCBI_API_KEY")

# --- Azure OpenAI Tool Schema Definition ---
tools_definition = [
    {
        "type": "function",
        "function": {
            "name": "esearch_ncbi",
            "description": "Performs a search on NCBI Eutils for a given term in a specified database (gene, snp, omim) and returns a list of UIDs.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "database": {
                        "type": "string",
                        "enum": ["gene", "snp", "omim"],
                        "description": "The NCBI database to search. Must be one of 'gene', 'snp', or 'omim'.",
                    },
                    "term": {
                        "type": "string",
                        "description": "The search term (e.g., 'BRCA1', 'rs12345', 'Alzheimer disease').",
                    },
                    "retmax": {
                        "type": "integer",
                        "description": "Maximum number of UIDs to return.",
                        "default": 5,
                    },
                },
                "required": ["database", "term"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "esummary_ncbi",
            "description": "Retrieves summaries for a list of UIDs from a specified NCBI Eutils database (gene, snp, omim). The structure of the summary varies by database.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "database": {
                        "type": "string",
                        "enum": ["gene", "snp", "omim"],
                        "description": "The NCBI database. Must be one of 'gene', 'snp', or 'omim'.",
                    },
                    "uids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "A list of NCBI UIDs obtained from esearch_ncbi.",
                    },
                    "retmax": {
                        "type": "integer",
                        "description": "Maximum number of summaries to return.",
                        "default": 5,
                    },
                },
                "required": ["database", "uids"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "efetch_ncbi",
            "description": "Retrieves full records for a list of UIDs from a specified NCBI Eutils database (gene, snp, omim). Can return large text data like FASTA or GenBank formats.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "database": {
                        "type": "string",
                        "enum": ["gene", "snp", "omim"],
                        "description": "The NCBI database. Must be one of 'gene', 'snp', or 'omim'.",
                    },
                    "uids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "A list of NCBI UIDs obtained from esearch_ncbi.",
                    },
                    "retmode": {
                        "type": "string",
                        "description": "Retrieval mode (e.g., 'text', 'xml'; 'json' if available for the db).",
                        "default": "text",
                    },
                    "rettype": {
                        "type": "string",
                        "description": "Retrieval type (e.g., 'fasta', 'gb' for nucleotide; varies by db).",
                        "default": "default",
                    },
                },
                "required": ["database", "uids"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "blast_put",
            "description": "Submits a sequence to NCBI BLAST. Returns a Request ID (RID) for polling results.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "sequence": {
                        "type": "string",
                        "description": "The DNA or protein sequence to BLAST.",
                    },
                    "program": {
                        "type": "string",
                        "enum": ["blastn", "blastp", "blastx", "tblastn", "tblastx"],
                        "description": "BLAST program. Common choices: 'blastn' (DNA-DNA), 'blastp' (protein-protein).",
                        "default": "blastn",
                    },
                    "database": {
                        "type": "string",
                        "enum": [
                            "nt",
                            "nr",
                            "refseq_rna",
                            "refseq_protein",
                            "swissprot",
                            "pdb",
                        ],
                        "description": "BLAST database. Common choices: 'nt' (nucleotide), 'nr' (non-redundant protein).",
                        "default": "nt",
                    },
                    "megablast": {
                        "type": "boolean",
                        "description": "Enable MEGABLAST for blastn (for highly similar sequences). Only applicable if program is 'blastn'.",
                        "default": True,
                    },
                    "hitlist_size": {
                        "type": "integer",
                        "description": "Number of aligned sequences to keep.",
                        "default": 10,
                    },
                },
                "required": ["sequence"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "blast_get",
            "description": "Retrieves BLAST results using a Request ID (RID). Important: This tool internally waits about 30 seconds before attempting to fetch results.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "rid": {
                        "type": "string",
                        "description": "The Request ID (RID) obtained from blast_put.",
                    },
                    "format_type": {
                        "type": "string",
                        "description": "Desired format of the results (e.g., 'Text', 'XML', 'JSON').",
                        "default": "Text",
                    },
                },
                "required": ["rid"],
                "additionalProperties": False,
            },
        },
    },
]

# --- NCBI E-utils Tool Definitions (Python Functions) ---


def esearch_ncbi(database: str, term: str, retmax: int = 5) -> str:
    """
    Performs a search on NCBI Eutils for a given term in a specified database.
    Returns a JSON string with a list of UIDs or an error.
    """
    print(
        f"TOOL EXECUTING: esearch_ncbi with database: {database}, term: {term}, retmax: {retmax}"
    )
    time.sleep(1)  # NCBI rate limiting courtesy
    try:
        params = {
            "db": database,
            "term": term,
            "retmax": retmax,
            "retmode": "json",
        }
        if NCBI_API_KEY:
            params["api_key"] = NCBI_API_KEY

        url = f"{NCBI_BASE_URL}esearch.fcgi"
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        if data.get("esearchresult", {}).get("idlist"):
            uids = data["esearchresult"]["idlist"]
            print(f"TOOL RESULT: esearch_ncbi found UIDs: {uids}")
            return json.dumps({"uids": uids})
        else:
            warning = (
                data.get("esearchresult", {})
                .get("warninglist", {})
                .get("phrasesnotfound", [])
            )
            error_detail = f"No UIDs found for term '{term}' in database '{database}'."
            if warning:
                error_detail += f" Phrases not found: {', '.join(warning)}"
            print(f"TOOL RESULT: esearch_ncbi: {error_detail}")
            return json.dumps({"error": error_detail, "uids": []})
    except requests.exceptions.RequestException as e:
        print(f"TOOL ERROR: esearch_ncbi failed: {e}")
        return json.dumps({"error": str(e)})
    except json.JSONDecodeError as e:
        print(f"TOOL ERROR: esearch_ncbi JSON decode failed: {e}")
        return json.dumps({"error": "Failed to parse NCBI response."})


def esummary_ncbi(database: str, uids: list[str], retmax: int = 5) -> str:
    """
    Retrieves summaries for a list of UIDs from a specified NCBI Eutils database.
    Returns a JSON string of the summaries or an error.
    """
    if not uids:
        return json.dumps({"error": "No UIDs provided for esummary_ncbi."})
    ids_str = ",".join(uids)
    print(
        f"TOOL EXECUTING: esummary_ncbi with database: {database}, UIDs: {ids_str}, retmax: {retmax}"
    )
    time.sleep(1)  # NCBI rate limiting courtesy
    try:
        params = {
            "db": database,
            "id": ids_str,
            "retmax": retmax,
            "retmode": "json",
        }
        if NCBI_API_KEY:
            params["api_key"] = NCBI_API_KEY

        url = f"{NCBI_BASE_URL}esummary.fcgi"
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        if "result" in data:
            # The structure of 'result' can vary. For 'gene', UIDs are keys.
            # For others, it might be a list or different structure.
            # This provides the raw result for the given UIDs.
            summaries = {
                k: v for k, v in data["result"].items() if k != "uids"
            }  # Exclude the 'uids' list if present
            if (
                not summaries
                and "uids" in data["result"]
                and not data["result"]["uids"]
            ):
                return json.dumps(
                    {
                        "error": f"No results found for UIDs: {ids_str} in database {database}."
                    }
                )

            print(
                f"TOOL RESULT: esummary_ncbi successful for UIDs: {ids_str} in database {database}"
            )
            return json.dumps(summaries)
        else:
            print(
                f"TOOL RESULT: esummary_ncbi found no summary for UIDs: {ids_str} in database {database}"
            )
            return json.dumps(
                {
                    "error": f"No summary found or unexpected format for UIDs {ids_str} in {database}"
                }
            )
    except requests.exceptions.RequestException as e:
        print(f"TOOL ERROR: esummary_ncbi failed: {e}")
        return json.dumps({"error": str(e)})
    except json.JSONDecodeError as e:
        print(f"TOOL ERROR: esummary_ncbi JSON decode failed: {e}")
        return json.dumps({"error": "Failed to parse NCBI response."})


def efetch_ncbi(
    database: str, uids: list[str], retmode: str = "text", rettype: str = "default"
) -> str:
    """
    Retrieves full records for a list of UIDs from a specified NCBI Eutils database.
    Returns a string of the fetched records (can be large) or a JSON error.
    """
    if not uids:
        return json.dumps({"error": "No UIDs provided for efetch_ncbi."})
    ids_str = ",".join(uids)
    print(
        f"TOOL EXECUTING: efetch_ncbi with database: {database}, UIDs: {ids_str}, retmode: {retmode}, rettype: {rettype}"
    )
    time.sleep(1)  # NCBI rate limiting courtesy
    try:
        params = {"db": database, "id": ids_str, "retmode": retmode}
        if rettype != "default":
            params["rettype"] = rettype
        if NCBI_API_KEY:
            params["api_key"] = NCBI_API_KEY

        url = f"{NCBI_BASE_URL}efetch.fcgi"
        response = requests.get(
            url, params=params, timeout=60
        )  # Increased timeout for potentially large data
        response.raise_for_status()
        # Efetch often returns plain text or XML, not JSON, so we return the text content directly
        content = response.text
        print(
            f"TOOL RESULT: efetch_ncbi successful for UIDs: {ids_str}. Content length: {len(content)}"
        )
        # Return as JSON string with content for consistency, or just content if LLM handles plain text.
        # For now, let's wrap it to make it clear it's a tool output.
        return json.dumps({"content": content[:20000]})  # Truncate if very large
    except requests.exceptions.RequestException as e:
        print(f"TOOL ERROR: efetch_ncbi failed: {e}")
        return json.dumps({"error": str(e)})


def blast_put(
    sequence: str,
    program: str = "blastn",
    database: str = "nt",
    megablast: bool = True,
    hitlist_size: int = 10,
) -> str:
    """
    Submits a sequence to NCBI BLAST.
    Returns a JSON string with the RID or an error.
    """
    print(
        f"TOOL EXECUTING: blast_put with sequence (first 30 chars): {sequence[:30]}..., program: {program}, database: {database}"
    )
    params = {
        "CMD": "Put",
        "PROGRAM": program,
        "DATABASE": database,
        "QUERY": sequence,
        "HITLIST_SIZE": str(hitlist_size),
    }
    if program == "blastn" and megablast:
        params["MEGABLAST"] = "on"

    try:
        response = requests.post(BLAST_BASE_URL, data=params, timeout=60)
        response.raise_for_status()
        # Extract RID from the HTML response
        match = re.search(r"RID = (\w+)", response.text)
        if match:
            rid = match.group(1)
            print(f"TOOL RESULT: blast_put successful. RID: {rid}")
            return json.dumps({"rid": rid})
        else:
            # Try to find QBlastInfo if available for more detailed error
            qblast_info_match = re.search(
                r"QBlastInfoBegin\s*Message=(.*)\s*QBlastInfoEnd",
                response.text,
                re.DOTALL,
            )
            if qblast_info_match:
                error_msg = qblast_info_match.group(1).strip()
                print(f"TOOL RESULT: blast_put failed. NCBI BLAST Error: {error_msg}")
                return json.dumps({"error": f"NCBI BLAST Error: {error_msg}"})

            print(
                f"TOOL RESULT: blast_put failed. Could not parse RID. Response: {response.text[:500]}"
            )
            return json.dumps({"error": "Could not parse RID from BLAST response."})
    except requests.exceptions.RequestException as e:
        print(f"TOOL ERROR: blast_put failed: {e}")
        return json.dumps({"error": str(e)})


def blast_get(rid: str, format_type: str = "Text") -> str:
    """
    Retrieves BLAST results using a Request ID (RID).
    Waits 30 seconds before attempting to fetch results, as per NCBI guidelines.
    Returns a JSON string with the BLAST report or an error.
    """
    print(f"TOOL EXECUTING: blast_get with RID: {rid}, format_type: {format_type}")
    print("Waiting 30 seconds for BLAST results as per NCBI guidelines...")
    time.sleep(30)  # Wait for BLAST to process

    params = {"CMD": "Get", "RID": rid, "FORMAT_TYPE": format_type}
    try:
        response = requests.get(
            BLAST_BASE_URL, params=params, timeout=120
        )  # Increased timeout
        response.raise_for_status()
        content = response.text

        # Check for "Status=WAITING" or "Status=UNKNOWN"
        if "Status=WAITING" in content or "Status=SEARCHING" in content:
            print(f"TOOL RESULT: blast_get for RID {rid} is still processing.")
            return json.dumps(
                {
                    "status": "WAITING",
                    "message": "BLAST job is still processing. Try again later.",
                }
            )
        if "Status=FAILED" in content or "Status=UNKNOWN" in content:
            error_msg = f"BLAST job for RID {rid} failed or status is unknown."
            # Try to extract a more specific message if possible
            message_match = re.search(r"Message=(.*)", content)
            if message_match:
                error_msg += f" NCBI Message: {message_match.group(1).strip()}"
            print(f"TOOL RESULT: {error_msg}")
            return json.dumps({"error": error_msg})

        print(
            f"TOOL RESULT: blast_get successful for RID: {rid}. Content length: {len(content)}"
        )
        return json.dumps({"report": content[:30000]})  # Truncate if very large
    except requests.exceptions.RequestException as e:
        print(f"TOOL ERROR: blast_get failed for RID {rid}: {e}")
        return json.dumps({"error": str(e)})


AVAILABLE_FUNCTIONS = {
    "esearch_ncbi": esearch_ncbi,
    "esummary_ncbi": esummary_ncbi,
    "efetch_ncbi": efetch_ncbi,
    "blast_put": blast_put,
    "blast_get": blast_get,
}
