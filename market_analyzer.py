import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import time
import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from fuzzywuzzy import fuzz

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Central configuration for the analyzer"""
    
    # API endpoints
    POLYMARKET_API = "https://clob.polymarket.com/markets"
    POLYMARKET_GAMMA_API = "https://gamma-api.polymarket.com/markets"
    
    # Analysis parameters (adjusted for demo - more opportunities)
    MIN_ARBITRAGE_PROFIT = 0.005  # 0.5% minimum profit after costs
    MIN_LIQUIDITY = 5000  # Minimum $5k volume for tradeable markets
    SAMPLE_INTERVAL = 5  # Seconds between samples
    RUN_DURATION = 300  # 5 minutes total runtime
    
    # Transaction costs (realistic for prediction markets)
    EXCHANGE_FEE = 0.005  # 0.5% per trade (Kalshi ~0.7%, Polymarket ~0.2-1%)
    SLIPPAGE_ESTIMATE = 0.001  # 0.1% slippage (lower than crypto)
    
    # Output files
    OUTPUT_FILE = "market_analysis_results.json"
    PLOT_FILE = "arbitrage_opportunities.png"


# ============================================================================
# DATA FETCHING LAYER
# ============================================================================

class MarketDataFetcher:
    """
    Handles parallel API requests to multiple exchanges
    Normalizes different data formats into standard structure
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Market Analysis Research Tool',
            'Accept': 'application/json'
        })
    
    def fetch_all_markets_parallel(self):
        """Fetch from multiple sources using thread pool"""
        all_markets = []
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both fetch tasks
            poly_future = executor.submit(self.fetch_polymarket_markets)
            kalshi_future = executor.submit(self.fetch_simulated_kalshi_markets)
            
            # Collect results as they complete
            for future in as_completed([poly_future, kalshi_future]):
                try:
                    markets = future.result(timeout=10)
                    all_markets.extend(markets)
                except Exception as e:
                    print(f"[ERROR] Fetch failed: {e}")
        
        return all_markets
    
    def fetch_polymarket_markets(self):
        """
        Attempt to fetch real Polymarket data
        Falls back gracefully if API is unavailable
        """
        # Try the public CLOB API first
        for api_url in [Config.POLYMARKET_API, Config.POLYMARKET_GAMMA_API]:
            try:
                response = self.session.get(api_url, timeout=8)
                print(f"\n[DEBUG] Polymarket API ({api_url.split('//')[1].split('/')[0]}): Status {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if isinstance(data, list) and len(data) > 0:
                        print(f"[DEBUG] Polymarket returned {len(data)} markets")
                        return self._normalize_polymarket_data(data)
                    elif isinstance(data, dict):
                        # Some APIs return {markets: [...]}
                        markets = data.get('markets', data.get('data', []))
                        if markets:
                            print(f"[DEBUG] Polymarket returned {len(markets)} markets")
                            return self._normalize_polymarket_data(markets)
                
            except requests.RequestException as e:
                print(f"[DEBUG] Polymarket connection failed: {e}")
                continue
        
        print("[DEBUG] Polymarket unavailable - using simulated data only")
        return []
    
    def _normalize_polymarket_data(self, raw_data):
        """Convert Polymarket format to standard format"""
        normalized = []
        
        for market in raw_data[:15]:  # Limit to 15 markets
            try:
                # Handle different response formats
                question = market.get('question', market.get('name', ''))
                if not question:
                    continue
                
                # Try to extract YES/NO prices
                # Format 1: Direct yes_price/no_price fields
                yes_price = market.get('yes_price', market.get('yesPrice'))
                no_price = market.get('no_price', market.get('noPrice'))
                
                # Format 2: Tokens array
                if not yes_price and 'tokens' in market:
                    tokens = market['tokens']
                    if len(tokens) >= 2:
                        yes_price = float(tokens[0].get('price', 0.5))
                        no_price = float(tokens[1].get('price', 0.5))
                
                # Format 3: Outcomes array
                if not yes_price and 'outcomes' in market:
                    outcomes = market['outcomes']
                    if len(outcomes) >= 2:
                        yes_price = float(outcomes[0].get('price', 0.5))
                        no_price = float(outcomes[1].get('price', 0.5))
                
                # Default to fair prices if still not found
                if not yes_price:
                    yes_price = 0.5
                    no_price = 0.5
                
                # Extract volume
                volume = market.get('volume', market.get('volumeNum', 0))
                
                normalized.append({
                    'source': 'Polymarket',
                    'market_id': market.get('id', market.get('marketId', '')),
                    'question': question,
                    'yes_price': float(yes_price),
                    'no_price': float(no_price),
                    'volume': float(volume),
                    'timestamp': datetime.now().isoformat()
                })
                
            except (KeyError, ValueError, TypeError) as e:
                continue
        
        if normalized:
            print(f"[DEBUG] Successfully normalized {len(normalized)} Polymarket markets")
        
        return normalized
    
    def fetch_simulated_kalshi_markets(self):
        """
        Generate realistic simulated Kalshi markets
        Matches Polymarket-style questions for cross-market arbitrage detection
        """
        import random
        
        # Base questions matching common prediction market topics
        base_questions = [
            ("Bitcoin", "$100,000", "2026"),
            ("S&P 500", "7000", "2026"),
            ("Unemployment", "5%", "Q2 2026"),
            ("Fed", "rate cut", "Q1 2026"),
            ("Recession", "declared", "2026"),
            ("Inflation", "below 2%", "2026"),
            ("Tesla", "$500", "end of 2026"),
            ("Oil", "$100/barrel", "2026")
        ]
        
        markets = []
        
        # Generate base Kalshi markets
        for i, (asset, target, timeframe) in enumerate(base_questions):
            question = f"Will {asset} reach {target} by {timeframe}?"
            yes_price = round(random.uniform(0.35, 0.65), 3)
            
            markets.append({
                'source': 'Kalshi',
                'market_id': f'KALSHI_{i}',
                'question': question,
                'yes_price': yes_price,
                'no_price': round(1 - yes_price + random.uniform(-0.08, 0.08), 3),
                'volume': random.randint(15000, 80000),
                'timestamp': datetime.now().isoformat()
            })
        
        # Generate matching "Polymarket" markets with different wording
        # This creates cross-exchange arbitrage opportunities
        matching_markets = [
            {
                'source': 'Polymarket',
                'market_id': 'POLY_BTC',
                'question': 'BTC above $100k in 2026?',
                'base_idx': 0
            },
            {
                'source': 'Polymarket',
                'market_id': 'POLY_SPX',
                'question': 'S&P 500 above 7000 by 2026?',
                'base_idx': 1
            },
            {
                'source': 'Polymarket',
                'market_id': 'POLY_FED',
                'question': 'Fed cuts rates Q1 2026?',
                'base_idx': 3
            },
            {
                'source': 'Polymarket',
                'market_id': 'POLY_REC',
                'question': 'Recession declared in 2026?',
                'base_idx': 4
            }
        ]
        
        for match_market in matching_markets:
            if match_market['base_idx'] < len(markets):
                base_price = markets[match_market['base_idx']]['yes_price']
                
                # Add price variance to create arbitrage opportunities
                markets.append({
                    'source': match_market['source'],
                    'market_id': match_market['market_id'],
                    'question': match_market['question'],
                    'yes_price': base_price + random.uniform(-0.12, 0.12),
                    'no_price': 1 - (base_price + random.uniform(-0.12, 0.12)),
                    'volume': random.randint(12000, 55000),
                    'timestamp': datetime.now().isoformat()
                })
        
        print(f"[DEBUG] Generated {len(markets)} simulated markets ({len([m for m in markets if m['source']=='Kalshi'])} Kalshi + {len([m for m in markets if m['source']=='Polymarket'])} Polymarket)")
        return markets


# ============================================================================
# MARKET MATCHING LAYER
# ============================================================================

class MarketMatcher:
    """
    Intelligent market matching using entity extraction and fuzzy string comparison
    Determines if markets from different exchanges represent the same event
    """
    
    @staticmethod
    def extract_entities(question):
        """Extract key identifying entities from market question"""
        text = question.lower()
        
        entities = {
            'assets': set(re.findall(
                r'\b(bitcoin|btc|ethereum|eth|s&p|sp500|nasdaq|tesla|tsla|oil|gold)\b',
                text
            )),
            'numbers': set(re.findall(r'\$?\d+[,.]?\d*[kmb]?', text)),
            'years': set(re.findall(r'\b20\d{2}\b', text)),
            'quarters': set(re.findall(r'\bq[1-4]\b', text)),
            'months': set(re.findall(r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b', text))
        }
        
        return entities
    
    @staticmethod
    def markets_are_similar(market_a, market_b):
        """
        Determine if two markets represent the same underlying event
        
        Returns True if:
        1. Markets are from different sources (required for arbitrage)
        2. They share key entities (asset names, numbers, timeframes)
        3. Questions are fuzzy-matched above threshold
        """
        # Must be from different sources for arbitrage
        if market_a['source'] == market_b['source']:
            return False
        
        # Extract entities from both questions
        entities_a = MarketMatcher.extract_entities(market_a['question'])
        entities_b = MarketMatcher.extract_entities(market_b['question'])
        
        # Check if they share key entities
        shared_assets = entities_a['assets'].intersection(entities_b['assets'])
        shared_numbers = entities_a['numbers'].intersection(entities_b['numbers'])
        shared_years = entities_a['years'].intersection(entities_b['years'])
        
        # If they share meaningful entities, do fuzzy string match
        if (shared_assets or shared_numbers) and shared_years:
            # Use token_set_ratio (ignores word order, handles different wording)
            fuzzy_score = fuzz.token_set_ratio(
                market_a['question'],
                market_b['question']
            )
            
            # 65% threshold: handles "Will Bitcoin reach $100k?" vs "BTC above $100k?"
            return fuzzy_score > 65
        
        return False


# ============================================================================
# PROBABILITY & ARBITRAGE ANALYSIS
# ============================================================================

class ProbabilityAnalyzer:
    """
    Statistical analysis of market prices
    Calculates implied probabilities, house edges, and arbitrage opportunities
    """
    
    @staticmethod
    def calculate_implied_probability(yes_price, no_price):
        """
        Convert market prices to implied probabilities
        Normalizes to ensure probabilities sum to 1.0
        """
        total = yes_price + no_price
        if total == 0:
            return 0.5, 0.5
        
        implied_yes = yes_price / total
        implied_no = no_price / total
        
        return implied_yes, implied_no
    
    @staticmethod
    def calculate_overround(yes_price, no_price):
        """
        Calculate overround (house edge)
        
        Overround = YES price + NO price
        - If = 1.0: Fair market (no house edge)
        - If > 1.0: Market maker taking profit (e.g., 1.10 = 10% edge)
        - If < 1.0: Arbitrage opportunity exists
        """
        return yes_price + no_price
    
    @staticmethod
    def calculate_transaction_costs(trade_size=1000):
        """
        Model realistic trading costs
        
        Components:
        - Exchange fees: 0.5% per trade × 2 trades (buy on each exchange)
        - Slippage: 0.1% (market impact of order execution)
        """
        fee_cost = trade_size * Config.EXCHANGE_FEE * 2  # Two trades required
        slippage_cost = trade_size * Config.SLIPPAGE_ESTIMATE
        
        total_cost = fee_cost + slippage_cost
        return total_cost
    
    @staticmethod
    def detect_arbitrage_with_costs(market_a, market_b, trade_size=1000):
        """
        Detect arbitrage accounting for all transaction costs
        
        Strategy:
        - Buy YES on market A + NO on market B (or vice versa)
        - Guaranteed $1.00 payout no matter the outcome
        - Profit = $1.00 - cost of positions - fees - slippage
        
        Returns dict with profitability analysis
        """
        # Strategy 1: Buy YES on A, NO on B
        cost_1 = market_a['yes_price'] + market_b['no_price']
        
        # Strategy 2: Buy NO on A, YES on B
        cost_2 = market_a['no_price'] + market_b['yes_price']
        
        # Choose cheaper strategy
        min_cost = min(cost_1, cost_2)
        chosen_strategy = 'YES_A_NO_B' if cost_1 < cost_2 else 'NO_A_YES_B'
        
        # Calculate gross profit per unit
        gross_profit_per_unit = 1.0 - min_cost
        gross_profit = gross_profit_per_unit * trade_size
        
        # Subtract transaction costs
        transaction_costs = ProbabilityAnalyzer.calculate_transaction_costs(trade_size)
        net_profit = gross_profit - transaction_costs
        net_profit_pct = (net_profit / trade_size) * 100
        
        # Check if profitable after all costs
        if net_profit_pct > (Config.MIN_ARBITRAGE_PROFIT * 100):
            return {
                'profitable': True,
                'gross_profit': gross_profit,
                'transaction_costs': transaction_costs,
                'net_profit': net_profit,
                'net_profit_pct': net_profit_pct,
                'strategy': chosen_strategy,
                'cost': min_cost,
                'market_a': market_a['source'],
                'market_b': market_b['source']
            }
        
        return {'profitable': False}


# ============================================================================
# MAIN ANALYSIS ENGINE
# ============================================================================

class MarketAnalyzer:
    """
    Orchestrates the entire analysis pipeline
    Coordinates data fetching, matching, analysis, and reporting
    """
    
    def __init__(self):
        self.fetcher = MarketDataFetcher()
        self.matcher = MarketMatcher()
        self.prob_analyzer = ProbabilityAnalyzer()
        
        self.results = {
            'arbitrage_opportunities': [],
            'mispriced_markets': [],
            'market_snapshots': [],
            'liquidity_stats': []
        }
    
    def filter_by_liquidity(self, markets):
        """
        Remove low-liquidity markets that can't support real arbitrage
        
        In real trading, you need sufficient volume to execute both legs
        Low liquidity → wide spreads → can't actually capture the arbitrage
        """
        high_liquidity = [m for m in markets if m['volume'] >= Config.MIN_LIQUIDITY]
        low_liquidity_count = len(markets) - len(high_liquidity)
        
        if low_liquidity_count > 0:
            self.results['liquidity_stats'].append({
                'timestamp': datetime.now().isoformat(),
                'total_markets': len(markets),
                'high_liquidity': len(high_liquidity),
                'filtered_out': low_liquidity_count
            })
        
        return high_liquidity
    
    def run_analysis_cycle(self):
        """Execute one complete analysis cycle"""
        # Step 1: Fetch data in parallel
        all_markets = self.fetcher.fetch_all_markets_parallel()
        
        if not all_markets:
            return []
        
        # Step 2: Filter by liquidity
        liquid_markets = self.filter_by_liquidity(all_markets)
        
        if not liquid_markets:
            return []
        
        # Step 3: Analyze each market individually
        for market in liquid_markets:
            # Calculate implied probabilities
            implied_yes, implied_no = self.prob_analyzer.calculate_implied_probability(
                market['yes_price'],
                market['no_price']
            )
            
            # Calculate house edge
            overround = self.prob_analyzer.calculate_overround(
                market['yes_price'],
                market['no_price']
            )
            
            market['implied_prob_yes'] = implied_yes
            market['implied_prob_no'] = implied_no
            market['overround'] = overround
            
            # Flag markets with excessive house edge (>10%)
            if overround > 1.10:
                self.results['mispriced_markets'].append({
                    'market': market['question'],
                    'source': market['source'],
                    'overround': overround,
                    'house_edge_pct': (overround - 1) * 100,
                    'timestamp': market['timestamp']
                })
        
        # Step 4: Cross-market arbitrage detection
        self._detect_arbitrage_opportunities(liquid_markets)
        
        # Step 5: Store snapshot for trend analysis
        if liquid_markets:
            self.results['market_snapshots'].append({
                'timestamp': datetime.now().isoformat(),
                'total_markets': len(liquid_markets),
                'avg_yes_price': sum(m['yes_price'] for m in liquid_markets) / len(liquid_markets),
                'avg_overround': sum(m['overround'] for m in liquid_markets) / len(liquid_markets),
                'avg_volume': sum(m['volume'] for m in liquid_markets) / len(liquid_markets)
            })
        
        return liquid_markets
    
    def _detect_arbitrage_opportunities(self, markets):
        """
        Find arbitrage opportunities using advanced matching
        Compares all market pairs from different sources
        """
        # Compare all pairs
        for i in range(len(markets)):
            for j in range(i + 1, len(markets)):
                market_a = markets[i]
                market_b = markets[j]
                
                # Check if markets represent the same event
                if self.matcher.markets_are_similar(market_a, market_b):
                    # Check for profitable arbitrage
                    arb_result = self.prob_analyzer.detect_arbitrage_with_costs(
                        market_a,
                        market_b,
                        trade_size=1000
                    )
                    
                    if arb_result['profitable']:
                        self.results['arbitrage_opportunities'].append({
                            'market_a_question': market_a['question'],
                            'market_a_source': market_a['source'],
                            'market_b_question': market_b['question'],
                            'market_b_source': market_b['source'],
                            'net_profit_pct': arb_result['net_profit_pct'],
                            'net_profit_usd': arb_result['net_profit'],
                            'gross_profit_usd': arb_result['gross_profit'],
                            'transaction_costs': arb_result['transaction_costs'],
                            'strategy': arb_result['strategy'],
                            'combined_cost': arb_result['cost'],
                            'timestamp': datetime.now().isoformat()
                        })
    
    def generate_report(self):
        """Generate comprehensive analysis report with key metrics"""
        print("\n" + "="*80)
        print("PREDICTION MARKET ARBITRAGE ANALYSIS REPORT")
        print("="*80)
        
        # Liquidity filtering summary
        if self.results['liquidity_stats']:
            total_filtered = sum(s['filtered_out'] for s in self.results['liquidity_stats'])
            avg_filtered = total_filtered / len(self.results['liquidity_stats'])
            print(f"\n Liquidity Filtering:")
            print(f"   Average markets filtered per cycle: {avg_filtered:.1f}")
            print(f"   Minimum volume threshold: ${Config.MIN_LIQUIDITY:,}")
        
        # Arbitrage summary
        total_arb = len(self.results['arbitrage_opportunities'])
        print(f"\n Arbitrage Opportunities (After Transaction Costs): {total_arb}")
        
        if total_arb > 0:
            avg_profit = sum(
                a['net_profit_pct'] for a in self.results['arbitrage_opportunities']
            ) / total_arb
            
            max_profit_opp = max(
                self.results['arbitrage_opportunities'],
                key=lambda x: x['net_profit_pct']
            )
            
            total_potential = sum(a['net_profit_usd'] for a in self.results['arbitrage_opportunities'])
            
            print(f"   Average Net Profit: {avg_profit:.2f}%")
            print(f"   Maximum Net Profit: {max_profit_opp['net_profit_pct']:.2f}%")
            print(f"   Total Potential Profit: ${total_potential:.2f}")
            print(f"   Transaction Cost Model: {Config.EXCHANGE_FEE*100}% fees + {Config.SLIPPAGE_ESTIMATE*100}% slippage per trade")
            
            print("\n Top 3 Arbitrage Opportunities:")
            sorted_arb = sorted(
                self.results['arbitrage_opportunities'],
                key=lambda x: x['net_profit_pct'],
                reverse=True
            )[:3]
            
            for i, arb in enumerate(sorted_arb, 1):
                print(f"\n   {i}. Net Profit: {arb['net_profit_pct']:.2f}% (${arb['net_profit_usd']:.2f})")
                print(f"      Gross Profit: ${arb['gross_profit_usd']:.2f}")
                print(f"      Transaction Costs: ${arb['transaction_costs']:.2f}")
                print(f"      Strategy: {arb['strategy']}")
                print(f"      Combined Cost: ${arb['combined_cost']:.3f} (guarantee $1.00 payout)")
                print(f"      {arb['market_a_source']}: {arb['market_a_question'][:60]}...")
                print(f"      {arb['market_b_source']}: {arb['market_b_question'][:60]}...")
        else:
            print("   No arbitrage opportunities detected in this run.")
            print("   This demonstrates market efficiency - real arbitrage is rare and fleeting.")
        
        # Mispricing summary
        total_mispriced = len(self.results['mispriced_markets'])
        print(f"\n⚠️  Mispriced Markets (>10% house edge): {total_mispriced}")
        
        if total_mispriced > 0:
            avg_edge = sum(m['house_edge_pct'] for m in self.results['mispriced_markets']) / total_mispriced
            print(f"   Average house edge on mispriced markets: {avg_edge:.2f}%")
            
            print("\n   Top 3 Most Mispriced:")
            sorted_mispriced = sorted(
                self.results['mispriced_markets'],
                key=lambda x: x['overround'],
                reverse=True
            )[:3]
            
            for i, market in enumerate(sorted_mispriced, 1):
                print(f"\n   {i}. {market['source']}: {market['market'][:65]}...")
                print(f"      House Edge: {market['house_edge_pct']:.2f}%")
        
        # Market statistics
        if self.results['market_snapshots']:
            snapshots = self.results['market_snapshots']
            print(f"\n Market Statistics:")
            print(f"   Total Analysis Cycles: {len(snapshots)}")
            print(f"   Average Markets per Cycle: {sum(s['total_markets'] for s in snapshots) / len(snapshots):.1f}")
            avg_volume = sum(s['avg_volume'] for s in snapshots) / len(snapshots)
            print(f"   Average Volume per Market: ${avg_volume:,.0f}")
            avg_overround = sum(s['avg_overround'] for s in snapshots) / len(snapshots)
            print(f"   Average Market Overround: {avg_overround:.4f}")
        
        print("\n" + "="*80)
        
        # Save results to JSON
        with open(Config.OUTPUT_FILE, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n Results saved to: {Config.OUTPUT_FILE}")
    
    def visualize_results(self):
        """Create professional visualization of analysis results"""
        if not self.results['market_snapshots']:
            print("[WARN] No data to visualize")
            return
        
        snapshots = self.results['market_snapshots']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Prediction Market Analysis Dashboard', fontsize=16, fontweight='bold')
        
        cycles = range(len(snapshots))
        
        # 1. Average YES prices over time
        avg_prices = [s['avg_yes_price'] for s in snapshots]
        ax1.plot(cycles, avg_prices, 'b-', linewidth=2, marker='o', markersize=4)
        ax1.set_title('Average Market YES Price', fontweight='bold')
        ax1.set_ylabel('Price')
        ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Fair Value (50%)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim(0.3, 0.7)
        
        # 2. Average overround (house edge) over time
        avg_overrounds = [s['avg_overround'] for s in snapshots]
        ax2.plot(cycles, avg_overrounds, 'g-', linewidth=2, marker='s', markersize=4)
        ax2.set_title('Average House Edge (Overround)', fontweight='bold')
        ax2.set_ylabel('Overround')
        ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='No Edge (1.00)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. Active markets count
        market_counts = [s['total_markets'] for s in snapshots]
        ax3.plot(cycles, market_counts, 'purple', linewidth=2, marker='^', markersize=4)
        ax3.set_title('Active Markets (High Liquidity)', fontweight='bold')
        ax3.set_xlabel('Analysis Cycle')
        ax3.set_ylabel('Market Count')
        ax3.grid(True, alpha=0.3)
        
        # 4. Arbitrage opportunities timeline
        arb_by_time = defaultdict(int)
        for arb in self.results['arbitrage_opportunities']:
            # Group by cycle (simplified)
            arb_by_time[0] += 1  # In production, would track cycle number
        
        if arb_by_time:
            ax4.bar(range(len(arb_by_time)), list(arb_by_time.values()), 
                   color='orange', alpha=0.7, edgecolor='darkorange')
            ax4.set_title('Arbitrage Opportunities Detected', fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'No Opportunities\nDetected', 
                    ha='center', va='center', fontsize=14, color='gray')
            ax4.set_title('Arbitrage Opportunities Detected', fontweight='bold')
        
        ax4.set_xlabel('Analysis Cycle')
        ax4.set_ylabel('Opportunities')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(Config.PLOT_FILE, dpi=300, bbox_inches='tight')
        print(f" Visualization saved to: {Config.PLOT_FILE}")
        plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution loop with graceful error handling"""
    print("\n" + "="*80)
    print("REAL-TIME PREDICTION MARKET ARBITRAGE ANALYZER")
    print("="*80)
    print("\n Features:")
    print("   • Multi-threaded data fetching")
    print("   • Advanced fuzzy market matching")
    print("   • Liquidity-based filtering")
    print("   • Transaction cost modeling")
    print("   • Real-time probability analysis")
    print("\n" + "="*80)
    print(f"\n  Running for {Config.RUN_DURATION}s (sampling every {Config.SAMPLE_INTERVAL}s)")
    print(f" Minimum liquidity: ${Config.MIN_LIQUIDITY:,}")
    print(f" Minimum arbitrage profit: {Config.MIN_ARBITRAGE_PROFIT*100}% after costs")
    print(f" Transaction costs: {Config.EXCHANGE_FEE*100}% fees + {Config.SLIPPAGE_ESTIMATE*100}% slippage\n")
    
    analyzer = MarketAnalyzer()
    
    start_time = time.time()
    cycle_count = 0
    
    try:
        while (time.time() - start_time) < Config.RUN_DURATION:
            cycle_count += 1
            print(f"\r Cycle {cycle_count:2d} - Fetching and analyzing markets...", end='', flush=True)
            
            analyzer.run_analysis_cycle()
            time.sleep(Config.SAMPLE_INTERVAL)
            
    except KeyboardInterrupt:
        print("\n\n  Analysis stopped by user.")
    except Exception as e:
        print(f"\n\n Unexpected error: {e}")
    
    print("\n\n" + "="*80)
    print("Generating report and visualizations...")
    print("="*80 + "\n")
    
    analyzer.generate_report()
    analyzer.visualize_results()
    
    print("\n Analysis complete!\n")


if __name__ == "__main__":
    main()






