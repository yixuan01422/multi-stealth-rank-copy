import os
import json
import shutil
import re
import requests
import time
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import argparse
from urllib.parse import quote, urljoin
from bs4 import BeautifulSoup
import random

class RealAmazonCollector:
    def __init__(self, category_name: str, output_dir: str = "data_new/amazon", image_size: int = 336):
        """Real Amazon product information collector"""
        self.category_name = category_name
        self.safe_name = category_name.replace(' ', '_').replace('-', '_')
        self.output_dir = Path(output_dir)
        self.image_size = (image_size, image_size)
        
        # Create directories
        self.temp_dir = Path("temp_downloads") / self.safe_name
        self.images_dir = self.output_dir / "images" / self.safe_name
        
        for dir_path in [self.temp_dir, self.output_dir, self.images_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Request headers - simulate real browser
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0'
        }
        
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        print(f"🚀 Initializing real crawler: {category_name}")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🖼️  Image directory: {self.images_dir}")
    
    def clean_filename(self, name: str) -> str:
        """Clean filename"""
        name = re.sub(r'[^\w\s-]', '', name)
        name = re.sub(r'\s+', '_', name)
        return name[:50]
    
    def get_name_prefix(self, name: str, n: int = 3) -> str:
        """Extract first n words of product name (case-insensitive)"""
        words = name.split()
        prefix_words = words[:min(len(words), n)]
        return ' '.join(prefix_words).lower()
    
    def get_product_links_from_search(self, target_count: int = 15) -> List[str]:
        """Get product links from search results"""
        print(f"🔍 Searching product links: {self.category_name}")
        
        try:
            search_url = f"https://www.amazon.com/s?k={quote(self.category_name)}&ref=sr_pg_1"
            time.sleep(random.uniform(2, 4))
            
            response = self.session.get(search_url, timeout=30)
            if response.status_code != 200:
                print(f"❌ Search failed: {response.status_code}")
                return []
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find product links
            product_links = []
            link_selectors = [
                'h2.a-size-mini a',
                'h2 a',
                '.s-product-image-container a',
                'a[href*="/dp/"]'
            ]
            
            for selector in link_selectors:
                links = soup.select(selector)
                for link in links:
                    href = link.get('href')
                    if href and '/dp/' in href:
                        full_url = urljoin('https://www.amazon.com', href)
                        # Clean URL, remove extra parameters
                        clean_url = re.sub(r'/ref=.*$', '', full_url)
                        if clean_url not in product_links:
                            product_links.append(clean_url)
                
                if len(product_links) >= target_count * 2:  # Collect enough links
                    break
            
            print(f"📦 Found {len(product_links)} product links")
            return product_links[:target_count * 2]  # Return 2x count to handle failures
            
        except Exception as e:
            print(f"💥 Search failed: {e}")
            return []
    
    def extract_product_details(self, product_url: str) -> Optional[Dict]:
        """Extract real information from product detail page"""
        try:
            print(f"  📄 Accessing product page...")
            time.sleep(random.uniform(3, 6))  # Increase delay to avoid being rate-limited
            
            response = self.session.get(product_url, timeout=30)
            if response.status_code != 200:
                print(f"  ❌ Access failed: {response.status_code}")
                return None
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract product title
            title = None
            title_selectors = [
                '#productTitle',
                '.product-title',
                'h1.a-size-large'
            ]
            
            for selector in title_selectors:
                elem = soup.select_one(selector)
                if elem:
                    title = elem.get_text(strip=True)
                    break
            
            if not title:
                return None
            
            print(f"  ✅ Product: {title[:50]}...")
            
            # Extract brand
            brand = self.extract_brand(soup, title)
            
            # Extract rating and review count
            rating, review_count = self.extract_rating_info(soup)
            
            # Extract price
            price = self.extract_price(soup)
            
            # Extract product features
            features = self.extract_features(soup)
            
            # Extract product description
            description = self.extract_description(soup)
            
            # Check description length, skip if too short
            if len(description) < 100:
                print(f"  ❌ Product description too short (length: {len(description)}), skipping")
                return None
            
            # Extract image URL
            image_url = self.extract_main_image(soup)
            
            if not image_url:
                print(f"  ❌ Product image not found")
                return None
            
            # Build rating string
            rating_str = "N/A"
            if rating and review_count:
                rating_str = f"{rating} ({review_count} reviews)"
            elif rating:
                rating_str = rating
            
            return {
                'name': title,
                'brand': brand,
                'rating': rating_str,
                'price': price,
                # 'features': features,
                'description': description,
                'image_url': image_url,
                'product_url': product_url
            }
            
        except Exception as e:
            print(f"  ❌ Failed to extract details: {e}")
            return None
    
    def extract_brand(self, soup, title: str) -> str:
        """Extract brand information"""
        # Try to extract from brand area
        brand_selectors = [
            '.po-brand .po-break-word',
            '#bylineInfo',
            '.author',
            'a[data-attribute="brand"]'
        ]
        
        for selector in brand_selectors:
            elem = soup.select_one(selector)
            if elem:
                brand_text = elem.get_text(strip=True)
                # Clean brand text
                brand = re.sub(r'^(Brand:|by|Visit the|Store)', '', brand_text, flags=re.IGNORECASE).strip()
                if brand and len(brand) > 1:
                    return brand
        
        # Extract brand from title (first word)
        words = title.split()
        if words:
            return words[0]
        
        return "Unknown Brand"
    
    def extract_rating_info(self, soup) -> tuple:
        """Extract rating and review count"""
        rating = None
        review_count = None
        
        # Rating
        rating_selectors = [
            '.a-icon-alt',
            '[data-hook="average-star-rating"] .a-icon-alt',
            '.cr-original-review-source-link .a-icon-alt'
        ]
        
        for selector in rating_selectors:
            elem = soup.select_one(selector)
            if elem:
                rating_text = elem.get('aria-label', '') or elem.get_text(strip=True)
                rating_match = re.search(r'(\d+\.?\d*)\s*out of', rating_text)
                if rating_match:
                    rating = rating_match.group(1)
                    break
        
        # Review count
        review_selectors = [
            '#acrCustomerReviewText',
            '[data-hook="total-review-count"]',
            '.cr-original-review-source-link'
        ]
        
        for selector in review_selectors:
            elem = soup.select_one(selector)
            if elem:
                review_text = elem.get_text(strip=True)
                review_match = re.search(r'([\d,]+)', review_text)
                if review_match:
                    review_count = review_match.group(1)
                    break
        
        return rating, review_count
    
    def extract_price(self, soup) -> str:
        """Extract price"""
        price_selectors = [
            '.a-price .a-offscreen',
            '#priceblock_dealprice',
            '#priceblock_ourprice',
            '.a-price-whole',
            '.a-price-range'
        ]
        
        for selector in price_selectors:
            elem = soup.select_one(selector)
            if elem:
                price_text = elem.get_text(strip=True)
                if price_text and '$' in price_text:
                    return price_text
        
        return "N/A"
    
    def extract_features(self, soup) -> List[str]:
        """Extract product features"""
        features = []
        
        # Feature area selectors
        feature_selectors = [
            '#feature-bullets ul li span',
            '.a-unordered-list.a-vertical li span',
            '[data-feature-name="featurebullets"] li',
            '.feature .a-list-item'
        ]
        
        for selector in feature_selectors:
            elements = soup.select(selector)
            for elem in elements:
                text = elem.get_text(strip=True)
                # Filter out useless text
                if (text and len(text) > 10 and 
                    not text.startswith('Make sure') and
                    not text.startswith('See more') and
                    'asin' not in text.lower() and
                    'item model number' not in text.lower()):
                    features.append(text)
            
            if features:
                break  # Stop if features found
        
        # Deduplicate and limit count
        unique_features = []
        for feature in features:
            if feature not in unique_features:
                unique_features.append(feature)
        
        return unique_features[:6]  # Return at most 6 features
    
    def extract_description(self, soup) -> str:
        """Extract product description"""
        desc_selectors = [
            '#feature-bullets ul',
            '#productDescription p',
            '.a-unordered-list.a-vertical',
            '[data-feature-name="productDescription"]'
        ]
        
        for selector in desc_selectors:
            elem = soup.select_one(selector)
            if elem:
                desc_text = elem.get_text(strip=True)
                if desc_text and len(desc_text) > 50:
                    # Clean description text
                    cleaned_desc = re.sub(r'\s+', ' ', desc_text)
                    return cleaned_desc  
        
        return f"High-quality {self.category_name} available on Amazon."
    
    def extract_main_image(self, soup) -> Optional[str]:
        """Extract main product image"""
        img_selectors = [
            '#landingImage',
            '.a-dynamic-image',
            '#imgBlkFront',
            '.image.item img'
        ]
        
        for selector in img_selectors:
            img = soup.select_one(selector)
            if img:
                src = img.get('data-old-hires') or img.get('data-a-dynamic-image') or img.get('src')
                if src:
                    # If dynamic image data, parse JSON
                    if src.startswith('{'):
                        try:
                            import json
                            img_data = json.loads(src)
                            # Select largest size image
                            max_size = 0
                            best_url = None
                            for url, size_str in img_data.items():
                                try:
                                    size = int(size_str.split(',')[0])
                                    if size > max_size:
                                        max_size = size
                                        best_url = url
                                except:
                                    continue
                            if best_url:
                                return best_url
                        except:
                            pass
                    elif src.startswith('http'):
                        return src
        
        return None
    
    def download_and_resize_image(self, url: str, filename: str) -> bool:
        """Download and resize image"""
        try:
            temp_path = self.temp_dir / filename
            response = self.session.get(url, timeout=30)
            
            if response.status_code != 200:
                return False
                
            with open(temp_path, 'wb') as f:
                f.write(response.content)
            
            # Resize image
            with Image.open(temp_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img.thumbnail(self.image_size, Image.Resampling.LANCZOS)
                new_img = Image.new('RGB', self.image_size, (255, 255, 255))
                
                paste_x = (self.image_size[0] - img.width) // 2
                paste_y = (self.image_size[1] - img.height) // 2
                new_img.paste(img, (paste_x, paste_y))
                
                final_path = self.images_dir / filename
                new_img.save(final_path, quality=95, format='JPEG')
            
            return True
            
        except Exception as e:
            print(f"  ❌ Image processing failed: {e}")
            return False
    
    def process_product(self, product_data: Dict, index: int) -> Optional[Dict]:
        """Process single product"""
        name = product_data['name']
        safe_name = self.clean_filename(name)
        image_filename = f"{index:03d}_{safe_name}.jpg"
        
        print(f"🖼️  Downloading image...")
        
        # Download and process image
        if not self.download_and_resize_image(product_data['image_url'], image_filename):
            print(f"  ❌ Image processing failed")
            return None
        
        # Build description
        description_parts = [f"Brand: {product_data['brand']}"]
        
        # if product_data['rating'] != 'N/A':
        #     description_parts.append(f"Rating: {product_data['rating']}")
        
        if product_data['price'] != 'N/A':
            description_parts.append(f"Price: {product_data['price']}")
        
        # if product_data['features']:
        #     key_features = product_data['features'][:3]
        #     features_text = "Key features: " + "; ".join(key_features)
        #     description_parts.append(features_text)
        
        description_parts.append(product_data['description'])
        
        # Natural field
        natural_desc = product_data['description']
        if len(natural_desc) > 150:
            natural_desc = natural_desc[:150] + "..."
        
        result = {
            'Name': name,
            'Description': ' | '.join(description_parts),
            # 'Natural': natural_desc,
            'image_path': f"{self.safe_name}/{image_filename}",
            # 'Brand': product_data['brand'],
            # 'Rating': product_data['rating'],
            # 'Features': product_data['features']
        }
        
        print(f"  ✅ Processing completed")
        return result
    
    def collect_real_products(self, target_count: int = 15) -> List[Dict]:
        """Collect real product data"""
        print(f"🎯 Target: {target_count} real {self.category_name} products")
        
        # Get product links
        product_links = self.get_product_links_from_search(target_count)
        
        if not product_links:
            print("❌ No product links found")
            return []
        
        print(f"🔗 Starting to access {len(product_links)} product pages...")
        
        # Extract product details
        processed_products = []
        downloaded_prefixes = []  # Record name prefixes of downloaded products
        
        for i, url in enumerate(tqdm(product_links, desc=f"Crawling {self.category_name}")):
            if len(processed_products) >= target_count:
                break
                
            print(f"\n📦 [{i+1}/{len(product_links)}] Processing product...")
            
            # Extract product details
            product_data = self.extract_product_details(url)
            if not product_data:
                continue
            
            # Check if product name is duplicate
            name_prefix = self.get_name_prefix(product_data['name'])
            if name_prefix in downloaded_prefixes:
                print(f"  ⚠️  Duplicate product name, skipping")
                continue
            
            # Process image and generate final data
            result = self.process_product(product_data, len(processed_products) + 1)
            if result:
                processed_products.append(result)
                downloaded_prefixes.append(name_prefix)
                print(f"  ✅ Successfully added product {len(processed_products)}")
        
        return processed_products
    
    def save_data(self, products: List[Dict]) -> None:
        """Save data"""
        output_file = self.output_dir / f"{self.safe_name}.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for product in products:
                json.dump(product, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"\n💾 Data saved: {output_file}")
        print(f"📊 Successfully collected: {len(products)} real products")
        print(f"🖼️  Images saved in: {self.images_dir}")
    
    def cleanup(self):
        """Clean up temporary files"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def run(self, target_count: int = 15):
        """Run real crawling process"""
        print(f"🚀 Starting real crawling for {self.category_name}")
        print("⚠️  Note: This will access real Amazon product pages")
        
        try:
            products = self.collect_real_products(target_count)
            
            if products:
                self.save_data(products)
                print(f"\n✅ Real crawling completed! Total {len(products)} products")
                return True
            else:
                print(f"\n❌ No products successfully crawled")
                return False
                
        finally:
            self.cleanup()


class MultiCategoryRealCollector:
    def __init__(self, output_dir: str = "data_new/amazon", image_size: int = 336):
        """Multi-category real crawler"""
        self.output_dir = output_dir
        self.image_size = image_size
        
        # self.category_groups = {
        #     # (Kitchen & Dining)
        #     'kitchen_appliances': ['coffee maker', 'blender', 'toaster', 'air fryer', 'rice cooker', 'food processor'],
        #     'kitchen_tools': ['knife set', 'cutting board', 'mixing bowls', 'measuring cups', 'can opener', 'peeler'],
        #     'cookware': ['non stick pan', 'cast iron skillet', 'stock pot', 'baking sheet', 'dutch oven', 'wok'],
        #     'dinnerware': ['dinner plates', 'coffee mug', 'wine glasses', 'flatware set', 'serving bowls', 'chopsticks'],
            
        #     # (Home & Kitchen)
        #     'home_decor': ['throw pillow', 'wall clock', 'picture frame', 'decorative vase', 'candle holder', 'wall art'],
        #     'bedding': ['bed sheets', 'comforter', 'pillow', 'mattress topper', 'blanket', 'duvet cover'],
        #     'bathroom': ['shower curtain', 'bath towels', 'toilet paper holder', 'soap dispenser', 'bath mat', 'toothbrush holder'],
        #     'storage': ['storage basket', 'shoe rack', 'closet organizer', 'plastic bins', 'drawer dividers', 'laundry hamper'],
        #     'cleaning': ['vacuum cleaner', 'mop', 'broom', 'cleaning spray', 'microfiber cloths', 'dustpan'],
            
        #     # (Electronics)
        #     'audio': ['wireless headphones', 'bluetooth speaker', 'earbuds', 'soundbar', 'portable speaker', 'noise cancelling headphones'],
        #     'phone_accessories': ['phone case', 'screen protector', 'phone charger', 'car phone mount', 'pop socket', 'charging cable'],
        #     'computer_accessories': ['keyboard', 'mouse', 'laptop stand', 'webcam', 'usb hub', 'laptop sleeve'],
        #     'cables': ['usb cable', 'hdmi cable', 'lightning cable', 'usb c cable', 'ethernet cable', 'extension cord'],
        #     'smart_home': ['smart bulb', 'smart plug', 'security camera', 'doorbell camera', 'smart thermostat', 'motion sensor'],
            
        #     # (Toys & Games)
        #     'toys': ['building blocks', 'action figure', 'puzzle', 'stuffed animal', 'remote control car', 'toy train'],
        #     'board_games': ['board game', 'card game', 'chess set', 'dominos', 'poker chips', 'dice set'],
        #     'outdoor_toys': ['water gun', 'frisbee', 'jump rope', 'bubble maker', 'kite', 'hula hoop'],
            
        #     # (Sports & Outdoors)
        #     'fitness': ['yoga mat', 'resistance bands', 'dumbbell set', 'foam roller', 'jump rope', 'exercise ball'],
        #     'sports_equipment': ['basketball', 'soccer ball', 'tennis racket', 'baseball glove', 'golf balls', 'ping pong paddle'],
        #     'outdoor_gear': ['camping tent', 'sleeping bag', 'hiking backpack', 'water bottle', 'flashlight', 'camping chair'],
        #     'cycling': ['bike helmet', 'bike lock', 'bike lights', 'cycling gloves', 'bike pump', 'water bottle holder'],
            
        #     # (Clothing, Shoes & Jewelry)
        #     'clothing': ['t shirt', 'hoodie', 'jeans', 'dress', 'jacket', 'sweater'],
        #     'shoes': ['running shoes', 'sneakers', 'sandals', 'boots', 'slippers', 'dress shoes'],
        #     'accessories': ['backpack', 'wallet', 'sunglasses', 'watch', 'belt', 'baseball cap'],
        #     'jewelry': ['necklace', 'earrings', 'bracelet', 'ring', 'anklet', 'brooch'],
            
        #     # (Beauty & Personal Care)
        #     'skincare': ['face mask', 'moisturizer', 'face wash', 'sunscreen', 'serum', 'toner'],
        #     'haircare': ['shampoo', 'conditioner', 'hair dryer', 'straightener', 'hair brush', 'hair oil'],
        #     'personal_care': ['electric toothbrush', 'razor', 'body wash', 'deodorant', 'nail clipper', 'tweezers'],
        #     'makeup': ['lipstick', 'foundation', 'mascara', 'eyeshadow palette', 'makeup brush set', 'eyeliner'],
            
        #     # (Health & Household)
        #     'health': ['thermometer', 'blood pressure monitor', 'first aid kit', 'hand sanitizer', 'face masks', 'vitamins'],
        #     'baby_products': ['baby stroller', 'car seat', 'baby monitor', 'diaper bag', 'baby bottle', 'pacifier'],
            
        #     # (Office Products)
        #     'office': ['notebook', 'pen set', 'desk organizer', 'stapler', 'tape dispenser', 'paper clips'],
        #     'desk_accessories': ['desk lamp', 'mouse pad', 'desk calendar', 'pencil holder', 'letter tray', 'desk mat'],
            
        #     # (Tools & Home Improvement)
        #     'tools': ['drill set', 'screwdriver set', 'hammer', 'tape measure', 'level', 'wrench set'],
        #     'hardware': ['light bulbs', 'batteries', 'duct tape', 'super glue', 'sandpaper', 'paint brush'],
            
        #     # (Automotive)
        #     'car_accessories': ['car phone mount', 'dash cam', 'car vacuum', 'tire pressure gauge', 'jumper cables', 'car air freshener'],
        #     'car_care': ['car wash soap', 'microfiber towels', 'tire shine', 'glass cleaner', 'wax', 'car sponge'],
            
        #     # (Patio, Lawn & Garden)
        #     'garden': ['garden hose', 'plant pots', 'garden gloves', 'watering can', 'pruning shears', 'garden rake'],
        #     'outdoor_living': ['patio furniture', 'grill', 'fire pit', 'outdoor umbrella', 'string lights', 'bird feeder'],
            
        #     # (Arts, Crafts & Sewing)
        #     'art_supplies': ['acrylic paint set', 'paint brushes', 'sketchbook', 'colored pencils', 'markers', 'canvas'],
        #     'crafts': ['glue gun', 'craft scissors', 'glitter', 'stickers', 'yarn', 'beads'],
        #     'sewing': ['sewing machine', 'thread', 'fabric scissors', 'pins', 'measuring tape', 'needles'],
            
        #     # (Pet Supplies)
        #     'pet_supplies': ['dog food', 'cat litter', 'pet bed', 'dog toy', 'pet brush', 'pet bowl'],
            
        #     # (Books & Media)
        #     'books': ['fiction book', 'cookbook', 'self help book', 'mystery novel', 'biography', 'children book'],
            
        #     # (Grocery & Gourmet Food)
        #     'snacks': ['protein bar', 'nuts', 'dried fruit', 'granola', 'popcorn', 'chips'],
        #     'beverages': ['coffee beans', 'tea', 'water bottle', 'protein powder', 'energy drink', 'juice']
        # }
        self.category_groups = {
                'audio': ['wireless headphones'],
                'cookware': ['non stick pan'],
                'makeup': ['lipstick'],
                'baby_products': ['baby stroller'],
                'desk_accessories': ['desk lamp']
            }
        
    def collect_category_group(self, group_name: str, per_category: int = 15):
        """Collect entire category group"""
        if group_name not in self.category_groups:
            print(f"❌ Unknown category group: {group_name}")
            return
        
        categories = self.category_groups[group_name]
        print(f"🎯 Starting real crawling for {group_name} category group")
        print(f"📋 Included categories: {categories}")
        print(f"⚠️  This will access approximately {len(categories) * per_category * 2} Amazon pages")
        print("="*60)
        
        success_count = 0
        total_products = 0
        
        for i, category in enumerate(categories, 1):
            print(f"\n🔄 [{i}/{len(categories)}] Crawling category: {category}")
            print("-" * 40)
            
            collector = RealAmazonCollector(
                category_name=category,
                output_dir=self.output_dir,
                image_size=self.image_size
            )
            
            if collector.run(per_category):
                success_count += 1
                jsonl_file = Path(self.output_dir) / f"{category.replace(' ', '_')}.jsonl"
                if jsonl_file.exists():
                    with open(jsonl_file, 'r', encoding='utf-8') as f:
                        total_products += len(f.readlines())
            
            time.sleep(5)  # Longer delay between categories
        
        print("\n" + "="*60)
        print(f"🎉 {group_name} real crawling completed!")
        print(f"✅ Successful categories: {success_count}/{len(categories)}")
        print(f"📊 Total products: {total_products}")


def main():
    parser = argparse.ArgumentParser(description='Real Amazon product crawler tool')
    parser.add_argument('--mode', type=str, choices=['single', 'group'], default='group',
                       help='Crawling mode')
    parser.add_argument('--category', type=str,
                       help='Single category name')
    parser.add_argument('--group', type=str,
                       help='Category group name')
    parser.add_argument('--count', type=int, default=15,
                       help='Number of products per category')
    parser.add_argument('--output_dir', type=str, default='data_new/amazon',
                       help='Output directory')
    parser.add_argument('--image_size', type=int, default=336,
                       help='Image size')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        if not args.category:
            print("❌ Need to specify --category parameter")
            return
        
        collector = RealAmazonCollector(
            category_name=args.category,
            output_dir=args.output_dir,
            image_size=args.image_size
        )
        collector.run(args.count)
        
    elif args.mode == 'group':
        if not args.group:
            print("❌ Need to specify --group parameter")
            return
        
        collector = MultiCategoryRealCollector(
            output_dir=args.output_dir,
            image_size=args.image_size
        )
        collector.collect_category_group(args.group, args.count)


if __name__ == "__main__":
    main()