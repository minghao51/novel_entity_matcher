"""Ingestion script for UNSPSC/MCC product categories."""

from typing import Any
import csv
import json
import requests

from .base import BaseFetcher, resolve_output_dirs
from novelentitymatcher.utils.logging_config import get_logger

logger = get_logger(__name__)


class UNSPSCFetcher(BaseFetcher):
    """Fetch UNSPSC product/service category codes."""

    SOURCE_URL = "https://unstats.un.org/unsd/services/v2/"

    FALLBACK_UNSPSC = [
        {"Code": "10", "Title": "Live animals and livestock"},
        {"Code": "11", "Title": "Crops"},
        {"Code": "12", "Title": "Forestry products"},
        {"Code": "13", "Title": "Fishing and hunting"},
        {"Code": "14", "Title": "Mining and ore dressing"},
        {"Code": "15", "Title": "Coal and petroleum"},
        {"Code": "16", "Title": "Crude petroleum and natural gas"},
        {"Code": "17", "Title": "Primary metal manufacturing"},
        {"Code": "18", "Title": "Food products"},
        {"Code": "19", "Title": "Beverage and tobacco products"},
        {"Code": "20", "Title": "Wood and paper products"},
        {"Code": "21", "Title": "Chemicals"},
        {"Code": "22", "Title": "Rubber and plastics"},
        {"Code": "23", "Title": "Non-metallic mineral products"},
        {"Code": "24", "Title": "Metal products"},
        {"Code": "25", "Title": "Machinery and equipment"},
        {"Code": "26", "Title": "Computer and office equipment"},
        {"Code": "27", "Title": "Electronic components"},
        {"Code": "28", "Title": "Communications equipment"},
        {"Code": "29", "Title": "Scientific and medical equipment"},
        {"Code": "30", "Title": "Transportation equipment"},
        {"Code": "31", "Title": "Furniture and fixtures"},
        {"Code": "32", "Title": "Paper and paper products"},
        {"Code": "33", "Title": "Printing and publishing"},
        {"Code": "34", "Title": "Textiles and apparel"},
        {"Code": "35", "Title": "Leather and leather products"},
        {"Code": "36", "Title": "Glass and glass products"},
        {"Code": "37", "Title": "Sporting and recreational equipment"},
        {"Code": "38", "Title": "Toys and games"},
        {"Code": "39", "Title": "Jewelry and precious metals"},
        {"Code": "40", "Title": "Financial and insurance services"},
        {"Code": "41", "Title": "Real estate services"},
        {"Code": "42", "Title": "Rental and leasing services"},
        {"Code": "43", "Title": "Professional services"},
        {"Code": "44", "Title": "Education services"},
        {"Code": "45", "Title": "Healthcare services"},
        {"Code": "46", "Title": "Food services"},
        {"Code": "47", "Title": "Accommodation services"},
        {"Code": "48", "Title": "Transportation services"},
        {"Code": "49", "Title": "Communication services"},
        {"Code": "50", "Title": "Wholesale trade services"},
        {"Code": "51", "Title": "Retail trade services"},
    ]

    def fetch(self) -> list[dict[str, Any]]:
        """Download UNSPSC codes from UNdata."""
        output_path = self.raw_dir / "unspsc.json"

        common_url = (
            "https://raw.githubusercontent.com/papermax/UNSPSC/master/UNSPSC_en.json"
        )

        if not output_path.exists():
            try:
                response = requests.get(common_url, timeout=30)
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "")
                if "json" not in content_type and "text" not in content_type:
                    raise ValueError(f"Unexpected Content-Type: {content_type}")
                if len(response.content) > 10 * 1024 * 1024:
                    raise ValueError("Response exceeds 10MB size limit")
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(response.text)
            except (requests.RequestException, ConnectionError, TimeoutError, ValueError) as e:
                logger.warning(f"UNSPSC fetch failed: {e}, using fallback data")
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(self.FALLBACK_UNSPSC, f)

        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if data else self.FALLBACK_UNSPSC

    def process(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert to standardized format."""
        entities = []

        for item in raw_data:
            code = str(item.get("Code", "")).strip()
            title = item.get("Title", "").strip() or item.get("Description", "").strip()

            if not code or not title:
                continue

            aliases = []
            segment = ""
            family = ""
            if len(code) >= 2:
                segment = code[:2]
                aliases.append(segment)
            if len(code) >= 4:
                family = code[:4]
                if family != segment:
                    aliases.append(family)
            if len(code) >= 6:
                class_code = code[:6]
                if class_code != family:
                    aliases.append(class_code)

            entities.append(
                {
                    "id": code,
                    "name": title,
                    "aliases": "|".join(aliases) if aliases else "",
                    "type": "product",
                    "system": "UNSPSC",
                }
            )

        return entities


class MCCFetcher(BaseFetcher):
    """Fetch MCC (Merchant Category Codes)."""

    SOURCE_URLS = [
        "https://raw.githubusercontent.com/greggles/mcc-codes/master/mcc_codes.csv"
    ]

    FALLBACK_MCC = [
        {"mcc": "0742", "description": "Veterinary Services"},
        {"mcc": "0780", "description": "Landscaping and Horticultural Services"},
        {
            "mcc": "1520",
            "description": "General Contractors - Residential and Commercial",
        },
        {"mcc": "1731", "description": "Electrical Contractors"},
        {"mcc": "1750", "description": "Carpentry Contractors"},
        {"mcc": "1761", "description": "Roofing and Siding Contractors"},
        {"mcc": "1771", "description": "Concrete Work Contractors"},
        {"mcc": "1799", "description": "Special Trade Contractors"},
        {"mcc": "2741", "description": "Miscellaneous Publishing and Printing"},
        {"mcc": "2796", "description": "Plates, Lithographers, Photoengravers"},
        {"mcc": "2842", "description": "Special Cleaning and Sanitation"},
        {"mcc": "3000", "description": "Airlines and Air Carriers"},
        {"mcc": "3029", "description": "Aircraft and Ship Building"},
        {"mcc": "3060", "description": "Motor Vehicle Supplies and New Parts"},
        {"mcc": "3070", "description": "Miscellaneous Automotive Equipment Rental"},
        {"mcc": "3080", "description": "Tire Retreading and Repair"},
        {"mcc": "3100", "description": "Mechanical Workers"},
        {"mcc": "3172", "description": "Leather Goods and Luggage"},
        {"mcc": "3199", "description": "Leather Products"},
        {"mcc": "3200", "description": "Furniture, Home Furnishings, and Equipment"},
        {"mcc": "3211", "description": "Wooden Floor Coverings"},
        {"mcc": "3220", "description": "China, Glassware, Metalware"},
        {"mcc": "3250", "description": "Floors, Carpet, and Rug Stores"},
        {"mcc": "3260", "description": "Electric Parts and Equipment"},
        {"mcc": "3270", "description": "Wall and Ceiling Materials"},
        {"mcc": "3280", "description": "Lumber and Other Building Materials"},
        {"mcc": "3290", "description": "Miscellaneous Home and Building Supply"},
        {"mcc": "3295", "description": "Paints, Varnishes, and Supplies"},
        {"mcc": "3300", "description": "Medical Equipment and Supplies"},
        {"mcc": "3500", "description": "Hotels and Motels"},
        {"mcc": "3512", "description": "Boat Dealers and Marinas"},
        {"mcc": "3520", "description": "Recreational Vehicles"},
        {"mcc": "3540", "description": "Hardware Stores"},
        {"mcc": "3560", "description": "General Merchandise Stores"},
        {"mcc": "3610", "description": "Department Stores"},
        {"mcc": "3620", "description": "Variety Stores"},
        {"mcc": "3630", "description": "Miscellaneous General Merchandise"},
        {"mcc": "3640", "description": "Grocery Stores"},
        {"mcc": "3650", "description": "Meat Markets"},
        {"mcc": "3660", "description": "Fruit and Vegetable Markets"},
        {"mcc": "3693", "description": "Video Tape Rental Stores"},
        {"mcc": "4011", "description": "Railroads"},
        {"mcc": "4119", "description": "Ambulance Services"},
        {"mcc": "4121", "description": "Limousines and Taxicabs"},
        {"mcc": "4214", "description": "Motor Freight Carriers"},
        {"mcc": "4225", "description": "Public Warehousing"},
        {"mcc": "4411", "description": "Steamship and Cruise Lines"},
        {"mcc": "4511", "description": "Airlines"},
        {"mcc": "4722", "description": "Travel Agencies"},
        {"mcc": "4723", "description": "Travel Agencies"},
        {"mcc": "4789", "description": "Transportation Services"},
        {"mcc": "4829", "description": "Money Orders and Wire Transfer"},
        {"mcc": "5200", "description": "Home Supply Warehouse Stores"},
        {"mcc": "5211", "description": "Lumber and Building Materials"},
        {"mcc": "5231", "description": "Paint and Wallpaper Stores"},
        {"mcc": "5251", "description": "Hardware Stores"},
        {"mcc": "5261", "description": "Nurseries and Garden Stores"},
        {"mcc": "5271", "description": "Mobile Home Dealers"},
        {"mcc": "5300", "description": "Wholesale Clubs"},
        {"mcc": "5310", "description": "Discount Stores"},
        {"mcc": "5311", "description": "Department Stores"},
        {"mcc": "5331", "description": "Variety Stores"},
        {"mcc": "5399", "description": "Miscellaneous General Merchandise"},
        {"mcc": "5411", "description": "Grocery Stores"},
        {"mcc": "5422", "description": "Freezer and Locker Meat Provisioners"},
        {"mcc": "5441", "description": "Candy, Nut, and Confectionery Stores"},
        {"mcc": "5451", "description": "Dairy Products Stores"},
        {"mcc": "5462", "description": "Bakeries"},
        {"mcc": "5499", "description": "Miscellaneous Food Stores"},
        {"mcc": "5511", "description": "Motor Vehicle Dealers"},
        {"mcc": "5521", "description": "Motor Vehicle Dealers - Used"},
        {"mcc": "5531", "description": "Automotive Parts and Accessories"},
        {"mcc": "5532", "description": "Automotive Tire Stores"},
        {"mcc": "5533", "description": "Automotive Parts and Accessories"},
        {"mcc": "5534", "description": "Tire Stores"},
        {"mcc": "5541", "description": "Service Stations"},
        {"mcc": "5542", "description": "Automated Fuel Dispensers"},
        {"mcc": "5551", "description": "Boat Dealers"},
        {"mcc": "5561", "description": "Camper Dealers"},
        {"mcc": "5571", "description": "Motorcycle Dealers"},
        {"mcc": "5592", "description": "Motor Home Dealers"},
        {"mcc": "5599", "description": "Miscellaneous Auto Dealers"},
        {"mcc": "5611", "description": "Men's and Boy's Clothing Stores"},
        {"mcc": "5621", "description": "Women's Ready-to-Wear Stores"},
        {"mcc": "5631", "description": "Women's Accessory and Specialty Shops"},
        {"mcc": "5641", "description": "Children's and Infant's Wear Stores"},
        {"mcc": "5651", "description": "Family Clothing Stores"},
        {"mcc": "5655", "description": "Sports Apparel Stores"},
        {"mcc": "5661", "description": "Shoe Stores"},
        {"mcc": "5681", "description": "Furriers and Fur Shops"},
        {"mcc": "5691", "description": "Men's and Women's Clothing Stores"},
        {"mcc": "5697", "description": "Tailors and Seamstresses"},
        {"mcc": "5698", "description": "Miscellaneous Apparel and Accessory Shops"},
        {"mcc": "5712", "description": "Furniture Stores"},
        {"mcc": "5713", "description": "Floor Covering Stores"},
        {"mcc": "5714", "description": "Drapery and Window Coverings"},
        {"mcc": "5715", "description": "China, Glassware, and Metalware Stores"},
        {"mcc": "5718", "description": "Fireplace and Fireplace Screens Stores"},
        {"mcc": "5722", "description": "Household Appliance Stores"},
        {"mcc": "5732", "description": "Electronics Stores"},
        {"mcc": "5733", "description": "Music Stores"},
        {"mcc": "5734", "description": "Computer Software Stores"},
        {"mcc": "5735", "description": "Record Stores"},
        {"mcc": "5811", "description": "Eating Places and Restaurants"},
        {"mcc": "5812", "description": "Eating Places and Restaurants"},
        {"mcc": "5813", "description": "Drinking Places"},
        {"mcc": "5814", "description": "Fast Food Restaurants"},
        {"mcc": "5912", "description": "Drug Stores and Pharmacies"},
        {"mcc": "5921", "description": "Package Stores - Beer, Wine, and Liquor"},
        {"mcc": "5931", "description": "Used Merchandise Stores"},
        {"mcc": "5932", "description": "Antique Shops"},
        {"mcc": "5933", "description": "Pawn Shops"},
        {"mcc": "5941", "description": "Sporting Goods Stores"},
        {"mcc": "5942", "description": "Book Stores"},
        {"mcc": "5943", "description": "Office, School Supply, and Stationery Stores"},
        {"mcc": "5944", "description": "Jewelry Stores"},
        {"mcc": "5945", "description": "Hobby, Toy, and Game Shops"},
        {"mcc": "5946", "description": "Camera and Photographic Supply Stores"},
        {"mcc": "5947", "description": "Gift, Card, Novelty, and Souvenir Shops"},
        {"mcc": "5948", "description": "Luggage and Leather Goods Stores"},
        {"mcc": "5949", "description": "Sewing, Needlework, and Fabric Stores"},
        {"mcc": "5950", "description": "Glass and Mirror Stores"},
        {"mcc": "5962", "description": "Direct Marketing - Travel"},
        {"mcc": "5963", "description": "Direct Marketing - Other"},
        {"mcc": "5964", "description": "Direct Marketing - Catalog Merchant"},
        {
            "mcc": "5965",
            "description": "Direct Marketing - Catalog and Direct Response",
        },
        {"mcc": "5966", "description": "Direct Marketing - Outbound Telemarketing"},
        {"mcc": "5967", "description": "Direct Marketing - Inbound Telemarketing"},
        {"mcc": "5969", "description": "Direct Marketing - Other Direct Marketing"},
        {"mcc": "5970", "description": "Artist Supply Stores"},
        {"mcc": "5971", "description": "Art Dealers and Galleries"},
        {"mcc": "5972", "description": "Stamp and Coin Stores"},
        {"mcc": "5973", "description": "Religious Goods Stores"},
        {"mcc": "5975", "description": "Hearing Aids Stores"},
        {"mcc": "5976", "description": "Orthopedic Goods Stores"},
        {"mcc": "5977", "description": "Cosmetic Stores"},
        {"mcc": "5983", "description": "Fuel Dealers"},
        {"mcc": "5992", "description": "Florists"},
        {"mcc": "5993", "description": "Tobacco Stores"},
        {"mcc": "5994", "description": "News Dealers and Newsstands"},
        {"mcc": "5995", "description": "Optical Goods Stores"},
        {"mcc": "5997", "description": "Furriers"},
        {"mcc": "5998", "description": "Wholesale Nurseries"},
        {"mcc": "5999", "description": "Miscellaneous and Specialty Retail Stores"},
    ]

    def fetch(self) -> list[dict[str, Any]]:
        """Download MCC codes."""
        output_path = self.raw_dir / "mcc_codes.csv"

        url = self.SOURCE_URLS[0]
        if not output_path.exists():
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "")
                if "csv" not in content_type and "text" not in content_type:
                    raise ValueError(f"Unexpected Content-Type: {content_type}")
                if len(response.content) > 5 * 1024 * 1024:
                    raise ValueError("Response exceeds 5MB size limit")
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(response.text)
            except (requests.RequestException, ConnectionError, TimeoutError, ValueError) as e:
                logger.warning(f"MCC fetch failed: {e}, using fallback data")
                with open(output_path, "w", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=["mcc", "description"])
                    writer.writeheader()
                    writer.writerows(self.FALLBACK_MCC)

        data = []
        with open(output_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)

        return data if data else self.FALLBACK_MCC

    def process(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert to standardized format."""
        entities = []

        for row in raw_data:
            code = row.get("mcc", "").strip() or row.get("MCC", "").strip()
            description = (
                row.get("description", "").strip()
                or row.get("Description", "").strip()
                or row.get("edited_description", "").strip()
                or row.get("combined_description", "").strip()
            )

            if not code or not description:
                continue

            aliases = []
            if len(code) >= 2:
                aliases.append(code[:2])

            entities.append(
                {
                    "id": code,
                    "name": description,
                    "aliases": "|".join(aliases),
                    "type": "product",
                    "system": "MCC",
                }
            )

        return entities


def run(raw_dir=None, processed_dir=None):
    """Execute product data ingestion."""
    raw_dir, processed_dir = resolve_output_dirs("products", raw_dir, processed_dir)

    fetcher = UNSPSCFetcher(raw_dir, processed_dir)
    fetcher.run("products_unspsc.csv")

    mcc_fetcher = MCCFetcher(raw_dir, processed_dir)
    mcc_fetcher.run("products_mcc.csv")


if __name__ == "__main__":
    run()
