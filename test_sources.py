"""Test script to verify source links functionality."""
import json

# Test the search tool output format
test_sources = [
    {
        "title": "US Census Bureau - Population Estimates",
        "url": "https://www.census.gov/programs-surveys/popest.html",
        "snippet": "The Southeast region has high population density and economic activity."
    },
    {
        "title": "Healthcare Cost Analysis 2024",
        "url": "https://example.com/healthcare-costs",
        "snippet": "Medical claims are higher in densely populated areas."
    }
]

test_summary = "The southeast region's high claim rate may be linked to its high population density and economic activity, as reported by the US Census Bureau."

# Test JSON format
output = json.dumps({
    "summary": test_summary,
    "sources": test_sources
})

print("Test Search Tool Output:")
print(output)
print("\n" + "="*80 + "\n")

# Parse it back
parsed = json.loads(output)
print("Parsed Summary:")
print(parsed["summary"])
print("\nParsed Sources:")
for idx, source in enumerate(parsed["sources"], 1):
    print(f"{idx}. {source['title']}")
    print(f"   URL: {source['url']}")
    print(f"   Snippet: {source['snippet']}")
    print()
