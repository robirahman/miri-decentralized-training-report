"""
Real-world internet latency (RTT) and bandwidth data compiled from measured sources.

Sources:
    - Azure Network Round-Trip Latency Statistics (P50, June 2025)
      https://learn.microsoft.com/en-us/azure/networking/azure-network-latency
    - AWS Inter-Region Latency via CloudPing.co (2025)
      https://www.cloudping.co/
    - Verizon Monthly IP Latency Data (Jan 2025 - Jan 2026)
      https://www.verizon.com/business/terms/latency/
    - Epsilon Telecom Global Network RTT Table
      https://epsilontel.com/global-network-footprint/network-latency-city/
    - Cloudflare Radar Internet Quality (2024)
      https://radar.cloudflare.com/quality
    - FCC Measuring Broadband America 13th Report (2024)
      https://www.fcc.gov/reports-research/reports/measuring-broadband-america/measuring-fixed-broadband-thirteenth-report
    - Hibernia Express Transatlantic Cable measurements
      https://www.thebroadcastbridge.com/content/entry/3988/trans-atlantic-network-latency-reduced
    - AWS Network Latency Concepts Blog
      https://aws.amazon.com/blogs/networking-and-content-delivery/network-latency-concepts-and-best-practices-for-a-resilient-architecture/

All RTT values are in milliseconds (ms).
All bandwidth values are in bits per second (bps).
"""

# =============================================================================
# LATENCY DATA: Round-Trip Time (RTT) in milliseconds
# =============================================================================

# --- Tier 1: Same Cloud Region / Metro Area (same city, <50 km) ---
# Sources: Azure intra-region, AWS intra-AZ, datacenter measurements

LATENCY_SAME_REGION_MS = {
    "description": "Within a single cloud region or metro area",

    # Same rack / same availability zone
    "same_az_typical_ms": 0.5,          # AWS: ~300us P50 intra-AZ, Azure: 65us-2ms
    "same_az_range_ms": (0.1, 2.0),     # Low: cluster placement group, High: cross-AZ same region

    # Measured cloud intra-region pairs (Azure P50, June 2025)
    "measured_pairs": {
        ("East US", "East US 2"):         10,   # Azure
        ("Australia Central", "Australia Central 2"): 3,  # Azure
        ("UK South", "UK West"):           7,   # Azure
        ("Japan East", "Japan West"):     12,   # Azure
        ("Korea Central", "Korea South"):  8,   # Azure
        ("UAE Central", "UAE North"):      6,   # Azure
        ("South Africa North", "South Africa West"): 20,  # Azure
        ("Norway East", "Norway West"):   10,   # Azure
        ("France Central", "France South"): 15,  # Azure
        ("Switzerland North", "Switzerland West"): 7,  # Azure
        ("Canada Central", "Canada East"): 14,   # Azure
        ("West US", "West US 2"):         25,   # Azure
        ("West US", "West US 3"):         20,   # Azure
    },

    # Epsilon Telecom measured (same metro)
    "epsilon_pairs": {
        ("New York", "New Jersey"):        4,   # Epsilon
        ("New York", "Ashburn"):          13,   # Epsilon (DC metro)
    },

    # AWS intra-AZ and cross-AZ
    "aws_intra_az_p50_us": 0.3,         # AWS blog: 300 microseconds
    "aws_cross_az_typical_ms": 1.0,      # AWS: typically 1-2ms cross-AZ

    # Representative values for modeling
    "typical_ms": 1.0,                   # Good default for same-region
    "range_ms": (0.1, 25.0),             # Full range observed
}


# --- Tier 2: Within a Continent (100-5000 km) ---
# Sources: Azure inter-region, AWS inter-region, Verizon, Epsilon

LATENCY_CONTINENTAL_MS = {
    "description": "Within a single continent (e.g., continental US, within Europe)",

    # ---- Continental US ----
    "us_pairs": {
        # Azure P50 (June 2025)
        ("East US", "West US"):            71,   # Azure: East US -> West US
        ("East US", "West US 2"):          68,   # Azure
        ("East US", "West US 3"):          59,   # Azure
        ("East US", "Central US"):         28,   # Azure
        ("East US", "South Central US"):   36,   # Azure
        ("East US", "North Central US"):   19,   # Azure
        ("East US", "West Central US"):    50,   # Azure
        ("North Central US", "South Central US"): 40, # Azure
        ("North Central US", "West US"):   51,   # Azure
        ("Central US", "West US"):         41,   # Azure
        ("South Central US", "West US"):   36,   # Azure
        ("South Central US", "West US 3"): 24,   # Azure (Dallas-Phoenix)
        ("East US", "Canada Central"):     20,   # Azure

        # AWS inter-region (CloudPing, 2025)
        ("us-east-1", "us-west-2"):       60,   # AWS: Virginia -> Oregon
        ("us-east-1", "us-east-2"):       13,   # AWS: Virginia -> Ohio

        # Epsilon Telecom measured
        ("New York", "Los Angeles"):       64,   # Epsilon
        ("New York", "Miami"):             40,   # Epsilon
        ("Ashburn", "Los Angeles"):        73,   # Epsilon
        ("Ashburn", "Miami"):              27,   # Epsilon
        ("Miami", "Los Angeles"):          60,   # Epsilon
    },

    # Verizon backbone averages (monthly, North America, 2025)
    "verizon_us_backbone_avg_ms": 33.7,  # Verizon: avg ~33.6-33.9ms throughout 2025

    # ---- Within Europe ----
    "europe_pairs": {
        # Azure P50 (June 2025)
        ("UK South", "France Central"):    11,   # Azure
        ("UK South", "West Europe"):       12,   # Azure (NL)
        ("UK South", "Germany West Central"): 17, # Azure
        ("UK South", "North Europe"):      13,   # Azure (Ireland)
        ("UK South", "Norway East"):       24,   # Azure
        ("UK South", "Sweden Central"):    37,   # Azure
        ("UK South", "Italy North"):       27,   # Azure
        ("France Central", "Germany West Central"): 12, # Azure
        ("France Central", "West Europe"): 13,   # Azure
        ("France Central", "Italy North"): 21,   # Azure
        ("France Central", "North Europe"): 20,  # Azure
        ("France Central", "Poland Central"): 29, # Azure
        ("Germany West Central", "Italy North"): 14, # Azure
        ("Germany West Central", "Poland Central"): 22, # Azure
        ("North Europe", "West Europe"):   18,   # Azure (Ireland-Netherlands)
        ("North Europe", "Norway East"):   28,   # Azure
        ("North Europe", "Sweden Central"): 34,  # Azure
        ("Germany North", "Poland Central"): 16,  # Azure

        # AWS inter-region (CloudPing, 2025)
        ("eu-west-1", "eu-central-1"):    21,   # AWS: Ireland -> Frankfurt
        ("eu-central-1", "eu-central-2"): 8,    # AWS: Frankfurt -> Zurich

        # Epsilon Telecom measured
        ("London", "Paris"):               11,   # Epsilon
        ("London", "Amsterdam"):           13,   # Epsilon
        ("London", "Frankfurt"):           20,   # Epsilon
        ("London", "Marseille"):           21,   # Epsilon (UK-S.France)
        ("Paris", "Frankfurt"):            11,   # Epsilon
        ("Paris", "Amsterdam"):            16,   # Epsilon
        ("Paris", "Marseille"):            12,   # Epsilon
        ("Frankfurt", "Amsterdam"):        10,   # Epsilon
        ("Frankfurt", "Milan"):            10,   # Epsilon
    },

    # Verizon backbone averages (monthly, Europe, 2025)
    "verizon_europe_backbone_avg_ms": 14.9, # Verizon: avg ~14.8-15.2ms throughout 2025

    # ---- Within Asia-Pacific ----
    "apac_pairs": {
        # Azure P50 (June 2025)
        ("Japan East", "Korea Central"):   29,   # Azure
        ("Japan East", "East Asia"):       53,   # Azure (Hong Kong)
        ("Japan East", "Southeast Asia"):  73,   # Azure (Singapore)
        ("East Asia", "Southeast Asia"):   36,   # Azure (HK-Singapore)
        ("East Asia", "Korea Central"):    41,   # Azure
        ("Australia East", "Australia Southeast"): 16, # Azure (Sydney-Melbourne)
        ("Australia East", "Japan East"):  104,  # Azure (Sydney-Tokyo)
        ("Southeast Asia", "Central India"): 56, # Azure (Singapore-India)
        ("Korea Central", "Korea South"):   8,   # Azure

        # AWS inter-region (CloudPing, 2025)
        ("ap-northeast-1", "ap-southeast-1"): 70, # AWS: Tokyo -> Singapore

        # Epsilon Telecom measured
        ("Singapore", "Kuala Lumpur"):      9,   # Epsilon
        ("Singapore", "Jakarta"):          16,   # Epsilon
        ("Singapore", "Hong Kong"):        39,   # Epsilon
        ("Singapore", "Tokyo"):            83,   # Epsilon
        ("Hong Kong", "Tokyo"):            46,   # Epsilon
        ("Hong Kong", "Jakarta"):          55,   # Epsilon
        ("Seoul", "Tokyo"):               22,   # Epsilon
        ("Sydney", "Singapore"):           95,   # Epsilon
    },

    # Verizon backbone averages (monthly, Asia-Pacific, 2025)
    "verizon_apac_backbone_avg_ms": 76.0,  # Verizon: ~65-88ms range, high variability

    # Representative values for modeling
    "typical_us_cross_country_ms": 65,    # NYC-LA type distance
    "typical_us_regional_ms": 30,          # NYC-Chicago type distance
    "typical_europe_ms": 15,               # London-Frankfurt type distance
    "typical_apac_ms": 60,                 # Tokyo-Singapore type distance
}


# --- Tier 3: Cross-Continental (5000-15000 km) ---
# Sources: Azure, AWS, Verizon, Epsilon, Hibernia Express

LATENCY_CROSS_CONTINENTAL_MS = {
    "description": "Between continents (e.g., US-Europe, US-Asia, Europe-Asia)",

    # ---- US to Europe (Transatlantic) ----
    "us_europe_pairs": {
        # Azure P50 (June 2025)
        ("East US", "UK South"):            79,  # Azure
        ("East US", "West Europe"):         85,  # Azure (Netherlands)
        ("East US", "France Central"):      88,  # Azure
        ("East US", "Germany West Central"): 94, # Azure
        ("East US", "North Europe"):        74,  # Azure (Ireland)
        ("East US", "Norway East"):         97,  # Azure
        ("East US", "Sweden Central"):     113,  # Azure
        ("West US", "UK South"):           147,  # Azure
        ("West US", "West Europe"):        153,  # Azure
        ("West US 2", "UK South"):         145,  # Azure
        ("West US 2", "North Europe"):     137,  # Azure
        ("Canada Central", "UK South"):     92,  # Azure
        ("Canada Central", "West Europe"):  98,  # Azure

        # AWS inter-region (CloudPing, 2025)
        ("us-east-1", "eu-west-1"):         70,  # AWS: Virginia -> Ireland
        ("us-east-1", "eu-central-1"):      93,  # AWS: Virginia -> Frankfurt
        ("us-west-2", "eu-west-1"):        118,  # AWS: Oregon -> Ireland

        # Epsilon Telecom measured
        ("London", "New York"):             75,  # Epsilon
        ("London", "Ashburn"):              84,  # Epsilon
        ("London", "Miami"):               111,  # Epsilon
        ("London", "Los Angeles"):         135,  # Epsilon
        ("Paris", "New York"):              78,  # Epsilon
        ("Frankfurt", "New York"):          85,  # Epsilon
    },

    # Verizon transatlantic backbone (monthly, 2025)
    "verizon_transatlantic_avg_ms": 70.1,  # Verizon: ~69.9-70.7ms throughout 2025

    # Hibernia Express (lowest-latency transatlantic cable)
    "hibernia_express_nyc_london_ms": 59,  # Hibernia Express: <58.95ms measured

    # ---- US to Asia (Transpacific) ----
    "us_asia_pairs": {
        # Azure P50 (June 2025)
        ("East US", "Japan East"):         164,  # Azure
        ("East US", "Korea Central"):      187,  # Azure
        ("East US", "East Asia"):          216,  # Azure (Hong Kong)
        ("East US", "Southeast Asia"):     224,  # Azure (Singapore)
        ("East US", "Central India"):      235,  # Azure
        ("West US", "Japan East"):         107,  # Azure
        ("West US", "Korea Central"):      130,  # Azure
        ("West US", "East Asia"):          159,  # Azure
        ("West US", "Southeast Asia"):     171,  # Azure (Singapore)
        ("West US 2", "Japan East"):       100,  # Azure
        ("West US 2", "Korea Central"):    123,  # Azure
        ("West US 2", "Southeast Asia"):   163,  # Azure
        ("West US 2", "East Asia"):        151,  # Azure (Hong Kong)
        ("Central US", "Japan East"):      136,  # Azure

        # AWS inter-region (CloudPing, 2025)
        ("us-east-1", "ap-northeast-1"):   152,  # AWS: Virginia -> Tokyo
        ("us-east-1", "ap-southeast-1"):   226,  # AWS: Virginia -> Singapore
        ("us-west-2", "ap-northeast-1"):   105,  # AWS: Oregon -> Tokyo
        ("us-west-2", "ap-southeast-1"):   161,  # AWS: Oregon -> Singapore

        # Epsilon Telecom measured
        ("Los Angeles", "Tokyo"):          100,  # Epsilon
        ("Los Angeles", "Hong Kong"):      144,  # Epsilon
        ("Los Angeles", "Singapore"):      181,  # Epsilon
        ("New York", "Tokyo"):             164,  # Epsilon
        ("New York", "Singapore"):         245,  # Epsilon
        ("Miami", "Tokyo"):               160,  # Epsilon
    },

    # Verizon transpacific backbone (monthly, 2025)
    "verizon_transpacific_avg_ms": 112.0,  # Verizon: ~111.7-112.2ms throughout 2025

    # ---- Europe to Asia ----
    "europe_asia_pairs": {
        # Azure P50 (June 2025)
        ("UK South", "Japan East"):        231,  # Azure
        ("UK South", "East Asia"):         187,  # Azure (Hong Kong)
        ("UK South", "Southeast Asia"):    155,  # Azure (Singapore)
        ("UK South", "Central India"):     129,  # Azure
        ("West Europe", "Japan East"):     235,  # Azure
        ("West Europe", "Southeast Asia"): 161,  # Azure
        ("France Central", "Japan East"):  214,  # Azure
        ("France Central", "Southeast Asia"): 148, # Azure
        ("France Central", "Central India"): 123, # Azure
        ("Germany West Central", "Japan East"): 219, # Azure
        ("Germany West Central", "Southeast Asia"): 157, # Azure

        # Epsilon Telecom measured
        ("London", "Tokyo"):               243,  # Epsilon
        ("London", "Singapore"):           164,  # Epsilon
        ("London", "Hong Kong"):           201,  # Epsilon
        ("Frankfurt", "Tokyo"):            245,  # Epsilon (est. from table)
        ("Frankfurt", "Singapore"):        162,  # Epsilon
    },

    # ---- US to South America ----
    "us_south_america_pairs": {
        # Azure P50 (June 2025)
        ("East US", "Brazil South"):       119,  # Azure
        ("South Central US", "Brazil South"): 141, # Azure
        ("West US", "Brazil South"):       176,  # Azure

        # AWS inter-region (CloudPing, 2025)
        ("us-east-1", "sa-east-1"):        115,  # AWS: Virginia -> Sao Paulo
    },

    # ---- US/Europe to Africa ----
    "to_africa_pairs": {
        # Azure P50 (June 2025)
        ("East US", "South Africa North"):  218, # Azure
        ("UK South", "South Africa North"): 161, # Azure
        ("France Central", "South Africa North"): 156, # Azure
        ("West Europe", "South Africa North"): 165, # Azure

        # Epsilon
        ("London", "Cape Town"):           146,  # Epsilon

        # AWS (CloudPing, 2025)
        ("us-east-1", "af-south-1"):       229,  # AWS: Virginia -> Cape Town
        ("eu-west-1", "af-south-1"):       156,  # AWS: Ireland -> Cape Town
    },

    # ---- US/Europe to Australia ----
    "to_australia_pairs": {
        # Azure P50 (June 2025)
        ("East US", "Australia East"):      199,  # Azure
        ("West US", "Australia East"):      141,  # Azure
        ("West US 2", "Australia East"):    162,  # Azure
        ("UK South", "Australia East"):     247,  # Azure
        ("France Central", "Australia East"): 241, # Azure

        # Epsilon
        ("Sydney", "Los Angeles"):          275,  # Epsilon (note: unusually high)
        ("Sydney", "Hong Kong"):            133,  # Epsilon
        ("Sydney", "Tokyo"):               175,  # Epsilon
    },

    # Representative values for modeling
    "typical_us_europe_ms": 75,            # NYC-London optimized
    "typical_us_asia_ms": 150,             # West Coast to Japan/Korea
    "typical_europe_asia_ms": 200,         # London-Tokyo
    "typical_us_south_america_ms": 120,    # East US to Sao Paulo
}


# --- Tier 4: Global Worst-Case (>15000 km, multi-hop routes) ---
# Sources: Azure, AWS, derived estimates

LATENCY_GLOBAL_WORST_CASE_MS = {
    "description": "Worst-case global routes (typically multi-hop, e.g., South America to SE Asia)",

    "worst_case_pairs": {
        # Azure P50 (June 2025) - verified long routes
        ("Brazil South", "Southeast Asia"):     332,  # Azure
        ("Brazil South", "Japan East"):         271,  # Azure
        ("Brazil South", "East Asia"):          321,  # Azure (Hong Kong)
        ("Brazil South", "Korea Central"):      294,  # Azure
        ("Brazil South", "Australia East"):     299,  # Azure
        ("Brazil South", "Central India"):      329,  # Azure
        ("Brazil South", "Indonesia Central"):  343,  # Azure
        ("Brazil South", "Malaysia West"):      336,  # Azure

        ("South Africa North", "Japan East"):   249,  # Azure
        ("South Africa North", "Southeast Asia"): 180, # Azure
        ("South Africa North", "Australia East"): 270, # Azure
        ("South Africa North", "Brazil South"):  319, # Azure
        ("South Africa North", "West US"):       274, # Azure

        ("Brazil South", "South Africa North"):  319, # Azure
        ("Brazil South", "Qatar Central"):       317, # Azure

        ("Australia East", "Brazil South"):      299, # Azure
        ("Australia East", "UK South"):          247, # Azure
        ("Australia East", "North Europe"):      256, # Azure
        ("Australia East", "Sweden Central"):    294, # Azure

        ("New Zealand North", "Brazil South"):   288, # Azure
        ("New Zealand North", "South Africa North"): 293, # Azure
        ("New Zealand North", "UK South"):       263, # Azure
    },

    # Representative values for modeling
    "typical_worst_case_ms": 300,           # e.g., Brazil-SE Asia
    "absolute_worst_case_ms": 350,          # Brazil-Indonesia type route
    "range_ms": (250, 400),                 # Practical range for global worst-case
}


# =============================================================================
# BANDWIDTH DATA: Available internet connection speeds
# =============================================================================

# All values in bits per second (bps)
Kbps = 1_000
Mbps = 1_000_000
Gbps = 1_000_000_000
Tbps = 1_000_000_000_000

BANDWIDTH_TIERS = {
    "description": "Typical internet bandwidth tiers available in 2024-2025",

    # --- Consumer Broadband ---
    "consumer_dsl": {
        "description": "DSL connections (legacy, declining availability)",
        "typical_download_bps": 25 * Mbps,
        "typical_upload_bps": 3 * Mbps,
        "range_download_bps": (10 * Mbps, 100 * Mbps),
        "range_upload_bps": (1 * Mbps, 10 * Mbps),
        "notes": "FCC 2024: DSL is slowest widely available tech; often fails to meet 100/20 Mbps broadband definition",
    },

    "consumer_cable": {
        "description": "Cable broadband (Spectrum, Xfinity, Cox, etc.)",
        "typical_download_bps": 300 * Mbps,
        "typical_upload_bps": 20 * Mbps,
        "range_download_bps": (100 * Mbps, 1200 * Mbps),  # DOCSIS 3.1 up to 1.2 Gbps
        "range_upload_bps": (5 * Mbps, 50 * Mbps),         # Cable upload is asymmetric
        "notes": "FCC 2024: 85% of cable subscribers exceed advertised speeds on download. "
                 "National avg download ~214 Mbps (all tech). Upload severely asymmetric.",
    },

    "consumer_fiber": {
        "description": "Residential fiber (AT&T Fiber, Google Fiber, Verizon Fios, etc.)",
        "typical_download_bps": 1 * Gbps,
        "typical_upload_bps": 1 * Gbps,       # Fiber is typically symmetric
        "range_download_bps": (200 * Mbps, 8 * Gbps),  # Google Fiber up to 8 Gbps
        "range_upload_bps": (200 * Mbps, 8 * Gbps),
        "common_tiers_bps": [
            200 * Mbps,    # Entry-level fiber
            500 * Mbps,    # Mid-tier
            1 * Gbps,      # Standard gigabit (most common)
            2 * Gbps,      # Premium tier
            5 * Gbps,      # AT&T 5 GIG
            8 * Gbps,      # Google Fiber 8 GIG (2024)
        ],
        "notes": "FCC 2024: 85% of fiber subscribers exceed advertised speeds. "
                 "Symmetric upload is the key differentiator vs cable.",
    },

    # --- Business / Commercial ---
    "business_broadband": {
        "description": "Business-grade cable/fiber with SLAs (Comcast Business, Spectrum Business, etc.)",
        "typical_download_bps": 500 * Mbps,
        "typical_upload_bps": 100 * Mbps,
        "range_download_bps": (100 * Mbps, 10 * Gbps),
        "range_upload_bps": (25 * Mbps, 10 * Gbps),
        "notes": "Higher SLAs, static IPs, priority support. Often shared infrastructure.",
    },

    "business_dedicated_internet": {
        "description": "Dedicated Internet Access (DIA) - uncontended, guaranteed bandwidth",
        "typical_bps": 1 * Gbps,              # Most common enterprise DIA tier
        "range_bps": (100 * Mbps, 100 * Gbps),
        "common_tiers_bps": [
            100 * Mbps,    # Small office
            500 * Mbps,    # Mid-size business
            1 * Gbps,      # Standard enterprise DIA
            10 * Gbps,     # Large enterprise / campus
            100 * Gbps,    # Hyperscale / data center interconnect
        ],
        "notes": "Symmetric, guaranteed bandwidth with SLA. Colt, Lumen, Verizon, etc. "
                 "Pricing ~$500-5000/mo for 1-10 Gbps depending on location.",
    },

    # --- Enterprise / Data Center ---
    "enterprise_leased_line": {
        "description": "Enterprise leased lines and private circuits",
        "typical_bps": 10 * Gbps,
        "range_bps": (1 * Gbps, 400 * Gbps),
        "common_tiers_bps": [
            1 * Gbps,       # Standard leased line
            10 * Gbps,      # Common enterprise
            25 * Gbps,      # High-capacity
            40 * Gbps,      # Metro Ethernet
            100 * Gbps,     # DC interconnect / backbone
            400 * Gbps,     # State-of-art backbone links (2024)
        ],
        "notes": "Private, dedicated circuits. Typically metro or regional. "
                 "100Gb leased lines: GBP 2000-8000/mo (UK, 2025).",
    },

    "cloud_interconnect": {
        "description": "Cloud provider direct connect / express route",
        "typical_bps": 10 * Gbps,
        "range_bps": (1 * Gbps, 100 * Gbps),
        "common_tiers_bps": [
            1 * Gbps,       # AWS Direct Connect standard
            10 * Gbps,      # AWS/Azure/GCP dedicated
            100 * Gbps,     # AWS Direct Connect max (2024)
        ],
        "notes": "AWS Direct Connect, Azure ExpressRoute, GCP Cloud Interconnect. "
                 "Lower and more consistent latency than public internet.",
    },

    "datacenter_internal": {
        "description": "Within-datacenter networking (NIC to NIC, switch fabric)",
        "typical_bps": 100 * Gbps,
        "range_bps": (25 * Gbps, 800 * Gbps),
        "common_tiers_bps": [
            25 * Gbps,      # Basic cloud VM NIC
            50 * Gbps,      # Enhanced networking
            100 * Gbps,     # InfiniBand HDR / 100GbE (common for GPU clusters)
            200 * Gbps,     # InfiniBand HDR200
            400 * Gbps,     # InfiniBand NDR (NVIDIA H100 clusters)
            800 * Gbps,     # InfiniBand NDR400 / next-gen (emerging 2025)
        ],
        "notes": "For GPU training clusters. H100 clusters typically use 400 Gbps InfiniBand NDR. "
                 "B200/GB200 clusters moving to 800 Gbps.",
    },
}


# =============================================================================
# CONVENIENCE: Summary tables for quick lookup
# =============================================================================

# Simplified latency tiers for modeling (representative RTT in ms)
LATENCY_TIERS_MS = {
    "same_rack":              0.2,      # Same rack, same AZ, placement group
    "same_az":                0.5,      # Same availability zone
    "cross_az_same_region":   1.5,      # Cross-AZ, same cloud region
    "same_metro":             5.0,      # Same city, different DCs
    "same_country_nearby":   20.0,      # e.g., NYC-Chicago, London-Paris
    "continental_us":        65.0,      # e.g., NYC-LA
    "within_europe":         15.0,      # e.g., London-Frankfurt
    "within_apac":           60.0,      # e.g., Tokyo-Singapore
    "transatlantic":         75.0,      # e.g., NYC-London
    "transpacific_west":    110.0,      # e.g., LA-Tokyo (West Coast)
    "transpacific_east":    165.0,      # e.g., NYC-Tokyo (East Coast)
    "us_to_india":          235.0,      # e.g., East US to Mumbai
    "us_to_singapore":      225.0,      # e.g., East US to Singapore
    "europe_to_tokyo":      235.0,      # e.g., London-Tokyo
    "europe_to_singapore":  160.0,      # e.g., London-Singapore
    "us_to_south_america":  120.0,      # e.g., East US to Sao Paulo
    "us_to_africa":         220.0,      # e.g., East US to Johannesburg
    "global_worst_case":    330.0,      # e.g., Brazil to SE Asia
}

# Simplified bandwidth tiers for modeling (in bps)
BANDWIDTH_TIERS_BPS = {
    "consumer_dsl":                25 * Mbps,
    "consumer_cable_download":    300 * Mbps,
    "consumer_cable_upload":       20 * Mbps,
    "consumer_fiber_symmetric":     1 * Gbps,
    "consumer_fiber_premium":       5 * Gbps,
    "business_broadband":         500 * Mbps,
    "business_dia_standard":        1 * Gbps,
    "business_dia_enterprise":     10 * Gbps,
    "enterprise_leased_line":      10 * Gbps,
    "enterprise_high_capacity":   100 * Gbps,
    "cloud_direct_connect":        10 * Gbps,
    "datacenter_gpu_cluster":     400 * Gbps,   # InfiniBand NDR (H100)
    "datacenter_next_gen":        800 * Gbps,   # InfiniBand NDR400 (B200/GB200)
}

# For WAN-based distributed training scenarios
# These represent the effective per-node bandwidth for inter-site communication
WAN_BANDWIDTH_SCENARIOS_BPS = {
    "home_participant_cable":       20 * Mbps,   # Typical cable upload
    "home_participant_fiber":        1 * Gbps,   # Symmetric fiber
    "university_lab":               10 * Gbps,   # Campus DIA
    "small_gpu_cluster":             1 * Gbps,   # Colo with DIA
    "enterprise_site":              10 * Gbps,   # Enterprise DIA
    "cloud_region_interconnect":   100 * Gbps,   # Cloud backbone inter-region
    "hyperscaler_backbone":          1 * Tbps,   # Google/Meta/AWS backbone links
}


# =============================================================================
# PHYSICAL CONSTANTS for latency estimation
# =============================================================================

PHYSICAL_CONSTANTS = {
    "speed_of_light_vacuum_km_s": 299_792,       # km/s
    "speed_of_light_fiber_km_s":  200_000,        # ~2/3 c in glass fiber
    "latency_per_1000km_fiber_ms": 5.0,           # ~5 ms one-way per 1000 km in fiber
    "rtt_per_1000km_fiber_ms":    10.0,            # ~10 ms RTT per 1000 km in fiber
    "typical_routing_overhead_factor": 1.5,        # Real paths are ~1.3-1.7x great circle
    "submarine_cable_median_rtt_ms": 70.76,        # Median RTT of submarine cable segments
    "submarine_cable_latency_per_1000km_rtt_ms": 10.0,  # ~10ms RTT per 1000km of cable
}


# =============================================================================
# GREAT CIRCLE DISTANCES (km) between major cities for estimation
# =============================================================================

CITY_DISTANCES_KM = {
    ("New York", "London"):         5_570,
    ("New York", "Los Angeles"):    3_944,
    ("New York", "Chicago"):        1_145,
    ("New York", "Tokyo"):         10_838,
    ("New York", "Singapore"):     15_345,
    ("New York", "Sao Paulo"):      7_680,
    ("New York", "Sydney"):        15_989,
    ("London", "Tokyo"):            9_566,
    ("London", "Singapore"):       10_843,
    ("London", "Frankfurt"):          636,
    ("London", "Cape Town"):        9_626,
    ("Los Angeles", "Tokyo"):       8_815,
    ("Los Angeles", "Singapore"):  14_114,
    ("Los Angeles", "Sydney"):     12_051,
    ("Sao Paulo", "Singapore"):    15_982,
    ("Sao Paulo", "Tokyo"):       18_553,
    ("Sao Paulo", "Cape Town"):     6_135,
    ("Sydney", "Singapore"):        6_288,
    ("Sydney", "Tokyo"):            7_823,
    ("Tokyo", "Singapore"):         5_310,
    ("Frankfurt", "Singapore"):    10_178,
}


def estimate_rtt_from_distance(distance_km, overhead_factor=1.5):
    """
    Estimate RTT from great-circle distance.

    Uses fiber speed of light (~200,000 km/s) and a routing overhead factor
    (typically 1.3-1.7x due to non-straight fiber paths and routing hops).

    Returns RTT in milliseconds.
    """
    fiber_speed = PHYSICAL_CONSTANTS["speed_of_light_fiber_km_s"]
    one_way_ms = (distance_km * overhead_factor / fiber_speed) * 1000
    return 2 * one_way_ms


def get_latency_for_scenario(scenario_name):
    """
    Get a representative RTT in ms for a named scenario.

    Parameters:
        scenario_name: One of the keys in LATENCY_TIERS_MS

    Returns:
        RTT in milliseconds
    """
    if scenario_name in LATENCY_TIERS_MS:
        return LATENCY_TIERS_MS[scenario_name]
    raise ValueError(
        f"Unknown scenario '{scenario_name}'. "
        f"Available: {list(LATENCY_TIERS_MS.keys())}"
    )


def get_bandwidth_for_tier(tier_name):
    """
    Get bandwidth in bps for a named tier.

    Parameters:
        tier_name: One of the keys in BANDWIDTH_TIERS_BPS

    Returns:
        Bandwidth in bits per second
    """
    if tier_name in BANDWIDTH_TIERS_BPS:
        return BANDWIDTH_TIERS_BPS[tier_name]
    raise ValueError(
        f"Unknown tier '{tier_name}'. "
        f"Available: {list(BANDWIDTH_TIERS_BPS.keys())}"
    )


if __name__ == "__main__":
    print("=" * 70)
    print("NETWORK LATENCY REFERENCE DATA")
    print("=" * 70)

    print("\n--- Latency Tiers (RTT in ms) ---")
    for scenario, rtt in sorted(LATENCY_TIERS_MS.items(), key=lambda x: x[1]):
        print(f"  {scenario:30s} : {rtt:8.1f} ms")

    print("\n--- Bandwidth Tiers ---")
    for tier, bw in BANDWIDTH_TIERS_BPS.items():
        if bw >= 1e12:
            print(f"  {tier:35s} : {bw/1e12:8.1f} Tbps")
        elif bw >= 1e9:
            print(f"  {tier:35s} : {bw/1e9:8.1f} Gbps")
        else:
            print(f"  {tier:35s} : {bw/1e6:8.1f} Mbps")

    print("\n--- Distance-based RTT Estimates ---")
    for (city1, city2), dist in sorted(CITY_DISTANCES_KM.items(), key=lambda x: x[1]):
        est_rtt = estimate_rtt_from_distance(dist)
        print(f"  {city1:15s} -> {city2:15s} : {dist:6d} km | est. {est_rtt:6.1f} ms RTT")
