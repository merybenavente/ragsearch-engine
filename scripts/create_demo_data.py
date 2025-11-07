#!/usr/bin/env python3
"""
Demo data script to populate the vector database with Ancient Roman engineering documents.
This script creates a library and adds documents about famous Roman engineering feats.

Usage examples:
  # Naive index (no parameters)
  poetry run python scripts/create_demo_data.py --index-type naive

  # LSH index with default parameters
  poetry run python scripts/create_demo_data.py --index-type lsh

  # LSH index with custom parameters
  poetry run python scripts/create_demo_data.py --index-type lsh --index-params '{"num_tables": 4, "num_hyperplanes": 2}'

  # VPTree index with default parameters (default)
  poetry run python scripts/create_demo_data.py --index-type vptree

  # VPTree index with custom parameters
  poetry run python scripts/create_demo_data.py --index-type vptree --index-params '{"leaf_size": 10}'
"""

import requests
from typing import List
import os
import argparse
import json

# Base URL for the API
BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

# Demo data about Ancient Roman engineering feats
ROMAN_ENGINEERING_DOCUMENTS = [
    {
        "title": "The Pantheon: Architectural Marvel",
        "content": """The Pantheon in Rome stands as one of the most remarkable architectural achievements of ancient Rome. Built around 126 AD during the reign of Emperor Hadrian, this temple dedicated to all Roman gods features the world's largest unreinforced concrete dome. The dome spans 43.3 meters (142 feet) in diameter and reaches the same height, creating a perfect sphere that would fit entirely within the building.

The construction techniques used were revolutionary for their time. The Romans used a special concrete mixture that became lighter toward the top of the dome, incorporating volcanic ash (pozzolan) from Mount Vesuvius, which gave the concrete its strength and durability. The dome's weight is reduced through a series of recessed panels called coffers, which not only serve a structural purpose but also create a stunning visual effect.

The oculus, a circular opening at the dome's apex measuring 8.2 meters across, provides the only source of natural light. This opening was intentionally left uncovered, allowing rain to enter and drain through carefully designed floor channels. The engineering precision required to create such a structure without modern tools demonstrates the advanced understanding of materials and structural forces possessed by Roman engineers.

The Pantheon's foundations extend deep into the Roman soil, with massive concrete blocks providing stability. The walls, constructed of brick and concrete, are over 6 meters thick at the base, tapering as they rise to support the dome's immense weight. This masterpiece has survived earthquakes, floods, and nearly 2,000 years of weather, testament to the superior engineering skills of its builders.""",
        "tags": ["architecture", "concrete", "dome", "Hadrian", "temple"],
    },
    {
        "title": "Roman Aqueducts: Engineering Water Systems",
        "content": """The Roman aqueduct system represents one of the most impressive engineering achievements in human history. Over the course of five centuries, Roman engineers constructed more than 400 aqueducts across the empire, bringing fresh water from distant sources to cities and towns. These structures utilized gravity flow, requiring precise calculations of gradients and elevations across vast distances.

The Aqua Claudia, completed in 52 AD, exemplifies Roman hydraulic engineering excellence. Stretching 69 kilometers from its source in the Anio valley to Rome, this aqueduct could deliver 200,000 cubic meters of water daily. The engineering challenge was immense: maintaining a consistent downward slope of just 0.34 meters per kilometer while crossing valleys, hills, and rivers.

Roman engineers employed various construction techniques depending on terrain. Underground channels, called specus, were carved through hills and mountains. Where valleys needed crossing, magnificent stone and concrete arches were built, such as the Pont du Gard in France, which rises 49 meters above the river. These arches distributed weight efficiently while maintaining the water channel's gradient.

The aqueduct system included settling tanks to remove sediment, distribution chambers to direct water to different areas, and sophisticated valve systems to control flow. Lead pipes, bronze fittings, and concrete-lined channels ensured water quality and system longevity. The engineering principles used by Romans—understanding hydraulics, materials science, and structural engineering—remained unsurpassed for over a millennium.

Quality control was paramount in aqueduct construction. Roman engineers used devices called chorobates and groma for surveying and leveling. Water quality was maintained through regular cleaning, inspection chambers, and careful source selection. The social impact was enormous: reliable water supply enabled urban growth, public baths, fountains, and improved sanitation.""",
        "tags": ["aqueducts", "hydraulics", "water", "Aqua Claudia", "Pont du Gard"],
    },
    {
        "title": "Roman Roads: The Empire's Circulatory System",
        "content": """Roman roads formed the backbone of the empire's transportation network, with over 400,000 kilometers of roads connecting every corner of the Roman world. The famous saying "all roads lead to Rome" reflected the reality of a road system that enabled rapid military movement, trade, and communication across vast distances.

The construction of Roman roads followed standardized engineering principles that ensured durability and efficiency. The typical Roman road consisted of four layers: the statumen (foundation of large stones), the rudus (layer of crushed stones and concrete), the nucleus (fine gravel and sand), and the summum dorsum (surface layer of fitted stones). This multi-layer construction distributed loads effectively and provided excellent drainage.

The Via Appia, begun in 312 BC, demonstrates Roman road engineering at its finest. Stretching 560 kilometers from Rome to Brindisi, this road featured precisely cut polygonal stones fitted so tightly that no mortar was needed. The road's surface was cambered to shed water, and drainage ditches ran alongside. Construction required moving massive amounts of earth and stone, bridging rivers, and cutting through hills.

Roman engineers developed innovative solutions for challenging terrain. In marshy areas, they built causeways on wooden pilings driven deep into the ground. Mountains were traversed through carefully graded switchbacks or tunnels. River crossings featured stone bridges built to withstand floods and heavy traffic. The engineering survey work was remarkably accurate, with roads maintaining consistent grades over long distances.

The roads served multiple purposes beyond transportation. They facilitated tax collection, postal systems, and military logistics. Milestones marked distances and provided traveler information. Way stations offered rest and provisions. The engineering legacy of Roman roads influenced construction methods for centuries, with many modern European highways following ancient Roman routes.""",
        "tags": ["roads", "Via Appia", "transportation", "infrastructure", "military"],
    },
    {
        "title": "Roman Sewers: The Cloaca Maxima",
        "content": """The Cloaca Maxima, Rome's great sewer system, represents one of the earliest and most sophisticated urban sanitation systems in history. Built in the 6th century BC under the Tarquin kings and expanded throughout the Roman period, this underground network drained the marshy valleys between Rome's hills and created the foundation for urban development.

The main sewer channel, constructed of massive stone blocks without mortar, measures up to 4 meters wide and 3 meters high. The engineering required precise calculation of water flow, gradients, and structural loads. The system utilized gravity flow, with carefully designed slopes directing wastewater from higher elevations toward the Tiber River outlet.

Roman engineers developed sophisticated branch networks feeding into the main sewer. Smaller channels collected water from streets, houses, and public buildings. The system included settling chambers, inspection points, and ventilation shafts. Construction required extensive excavation in difficult conditions, with workers operating in confined spaces while maintaining structural integrity.

The engineering challenges were immense. The sewer had to handle not only human waste but also rainwater runoff from the entire city. Roman engineers designed overflow channels and retention areas to prevent flooding during heavy rains. The system's capacity was carefully calculated based on population density and water usage patterns.

Maintenance was crucial for system operation. Roman engineers developed techniques for cleaning blockages, repairing damaged sections, and inspecting the network. Special crews, called cloacarii, were responsible for maintenance work. The system's design allowed for periodic cleaning and repair without disrupting city operations.

The Cloaca Maxima's success enabled Rome's growth into a city of over one million inhabitants. Without effective sanitation, urban density on this scale would have been impossible. The engineering principles developed for this system influenced urban planning throughout the empire and established standards for public health infrastructure.""",
        "tags": [
            "sewers",
            "sanitation",
            "Cloaca Maxima",
            "urban planning",
            "public health",
        ],
    },
    {
        "title": "Roman Military Engineering: Siege Weapons and Fortifications",
        "content": """Roman military engineering combined practical construction skills with innovative tactical applications. Roman armies were essentially mobile engineering units, capable of building roads, bridges, fortifications, and siege weapons wherever campaigns took them. This engineering capability was fundamental to Roman military success.

The Roman ballista represents sophisticated mechanical engineering applied to warfare. These torsion-powered artillery pieces could accurately launch projectiles over 400 meters. The engineering required understanding of materials science, mechanical advantage, and projectile physics. Roman engineers developed standardized designs that could be rapidly assembled and maintained in field conditions.

Siege engineering reached its peak with Roman innovations. The agger, a massive earthwork ramp, allowed armies to breach city walls by creating an inclined approach. The construction required moving enormous quantities of earth and stone, often under enemy fire. Roman engineers developed techniques for rapid construction using available materials and slave labor.

Roman castra (military camps) demonstrated systematic engineering applied to temporary structures. Every camp followed standardized layouts with precise measurements and construction techniques. Engineers surveyed sites, laid out streets, and supervised construction of walls, gates, and buildings. The engineering efficiency enabled Roman armies to create defensible positions quickly.

The siege of Masada (73-74 AD) showcases Roman engineering prowess. Faced with an impregnable fortress, Roman engineers constructed a massive siege ramp using an estimated 11,000 tons of stone and earth. The ramp, still visible today, demonstrates the engineering capability to overcome any obstacle through systematic application of construction principles.

Roman engineers also developed sophisticated bridge-building techniques. Caesar's bridge across the Rhine, constructed in just 10 days, featured precisely calculated timber construction capable of supporting entire legions. The engineering required understanding of river hydraulics, load distribution, and construction logistics.""",
        "tags": ["military", "siege", "ballista", "fortifications", "bridges"],
    },
    {
        "title": "Roman Concrete: Revolutionary Building Material",
        "content": """Roman concrete revolutionized construction and enabled architectural achievements that remained unsurpassed for centuries. Unlike modern concrete, Roman concrete was made from lime, volcanic ash (pozzolan), and aggregate. This mixture created a material that actually became stronger over time, especially when exposed to seawater.

The discovery of pozzolan's properties near Mount Vesuvius was crucial to Roman engineering success. This volcanic ash, when mixed with lime, created a hydraulic cement that could set underwater. Roman engineers systematically exploited this material, establishing quarries and developing quality control methods to ensure consistent performance.

Roman concrete construction techniques were remarkably sophisticated. Engineers understood the importance of aggregate gradation, using different stone sizes for different applications. They developed methods for placing concrete in difficult conditions, including underwater construction for harbor works. The concrete was often reinforced with bronze clamps or wooden elements.

The Pantheon's dome showcases Roman concrete engineering at its finest. The concrete mixture varies throughout the structure, with heavier aggregate at the base and lighter pumice near the top. This engineering innovation reduced the dome's weight while maintaining structural integrity. The construction required precise planning and skilled execution.

Roman engineers also developed specialized concrete applications. Marine concrete, used for harbor construction, incorporated specific volcanic materials that resisted seawater corrosion. The harbors at Caesarea and Portus featured massive concrete blocks placed underwater, creating artificial harbors that facilitated trade throughout the Mediterranean.

Quality control in concrete production was essential. Roman engineers developed testing methods to ensure proper mixture ratios and curing conditions. They understood the importance of aggregate selection, water quality, and construction timing. This systematic approach to materials science enabled consistent results across the empire.

The durability of Roman concrete is legendary. Many structures built 2,000 years ago remain standing today, testament to the engineering knowledge and construction quality achieved by Roman builders. Modern analysis reveals that Roman concrete continues to strengthen through chemical reactions with seawater and atmospheric conditions.""",
        "tags": [
            "concrete",
            "pozzolan",
            "materials science",
            "construction",
            "durability",
        ],
    },
]


def create_library(name: str, index_type: str, index_params: dict = None) -> str:
    """Create a new library and return its ID."""
    library_data = {
        "name": name,
        "username": "demo_user",
        "tags": ["demo", "ancient_rome", "engineering"],
        "index_type": index_type,
    }

    if index_params:
        library_data["index_params"] = index_params

    response = requests.post(f"{BASE_URL}/libraries/", json=library_data)
    response.raise_for_status()
    return response.json()["id"]


def add_document(library_id: str, title: str, content: str, tags: List[str]) -> str:
    """Add a document to the library and return its ID."""
    document_data = {
        "text": content,
        "username": "demo_user",
        "tags": tags + ["demo", f"title:{title}"],
        "chunk_size": 100,
    }

    response = requests.post(
        f"{BASE_URL}/libraries/{library_id}/documents/", json=document_data
    )
    response.raise_for_status()
    return response.json()["id"]


def test_search(library_id: str, query: str) -> None:
    """Test the search functionality with a sample query."""
    search_data = {"query_text": query, "k": 5}

    response = requests.post(
        f"{BASE_URL}/libraries/{library_id}/search", json=search_data
    )
    response.raise_for_status()

    results = response.json()
    print(f"\nSearch Results for '{query}':")
    print(
        f"Found {len(results['results'])} results out of {results['total_chunks_searched']} chunks"
    )
    print(f"Query time: {results['query_time_ms']:.2f}ms")

    for i, result in enumerate(results["results"], 1):
        chunk = result["chunk"]
        score = result["similarity_score"]
        preview = (
            chunk["text"][:100] + "..." if len(chunk["text"]) > 100 else chunk["text"]
        )
        print(f"{i}. Score: {score:.3f} - {preview}")


def main():
    """Main function to populate the database with demo data."""
    parser = argparse.ArgumentParser(description="Create demo data for vector database")
    parser.add_argument(
        "--index-type",
        default="vptree",
        choices=["naive", "lsh", "vptree"],
        help="Index type to use (default: vptree)",
    )
    parser.add_argument(
        "--index-params",
        type=str,
        help='Index parameters as JSON string (e.g., \'{"leaf_size": 10}\' for vptree, \'{"num_tables": 8, "num_hyperplanes": 6}\' for lsh)',
    )

    args = parser.parse_args()

    print("Creating Roman Engineering Demo Database...")
    print(f"Using index type: {args.index_type}")

    index_params = None
    if args.index_params:
        try:
            index_params = json.loads(args.index_params)
            print(f"Using index params: {index_params}")
        except json.JSONDecodeError as e:
            print(f"Error parsing index params: {e}")
            return

    try:
        # Create library
        library_id = create_library(
            "Ancient Roman Engineering Feats", args.index_type, index_params
        )
        print(f"Created library with ID: {library_id}")

        # Add documents
        document_ids = []
        for doc in ROMAN_ENGINEERING_DOCUMENTS:
            doc_id = add_document(library_id, doc["title"], doc["content"], doc["tags"])
            document_ids.append(doc_id)
            print(f"Added document: {doc['title']} (ID: {doc_id})")

        print(f"\nSuccessfully added {len(document_ids)} documents to the library!")

        # Test searches
        test_queries = [
            "concrete construction techniques",
            "aqueduct water systems",
            "Roman roads construction",
            "military engineering",
            "architectural dome design",
        ]

        for query in test_queries:
            test_search(library_id, query)

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        print("Make sure the vector database API is running on http://localhost:8000")
        print("Start it with: poetry run uvicorn src.vector_db.api.main:app --reload")


if __name__ == "__main__":
    main()
