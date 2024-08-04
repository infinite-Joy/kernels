from crewai_tools import BaseTool


class SearchCSV(BaseTool):
    name: str = "Search a CSV's content"
    description: str = (
        "Search through the CSV content to find the "
        "best wedding venue"
    )

    def _run(self, country: str, style: str, budget: int) -> str:
        # Implementation goes here
        print()
        print('Search CSV argument')
        print(country, style, budget)
        return "name:Kayumanis Nusa Dua Private Villa & Spa,place:Nusa Dua,price:8,000 - 18,000 dollars"
