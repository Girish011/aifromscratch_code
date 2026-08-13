"""Shared mock sales tools for week2 function-calling / MCP notebook.

Import from any cell without re-running the notebook top cell:

    from sales_tools import tools, TOOL_IMPL, get_product_price, get_customer_discount
"""

PRODUCT_DB = {
    "starter": 99,
    "professional": 299,
    "enterprise": "custom (contact sales)",
}

CUSTOMER_DB = {
    "acme_corp": {"tier": "enterprise", "discount": 0.15},
    "startup_inc": {"tier": "starter", "discount": 0.0},
}


def get_product_price(product_name: str) -> str:
    """Return the price of a product plan (monthly)."""
    product_name = product_name.lower()
    if product_name in PRODUCT_DB:
        price = PRODUCT_DB[product_name]
        return f"The {product_name} plan costs ${price} per month."
    return "Product not found."


def get_customer_discount(company: str) -> str:
    """Return the discount and tier for a customer company."""
    company = company.lower()
    if company in CUSTOMER_DB:
        info = CUSTOMER_DB[company]
        pct = info["discount"] * 100
        return f"{company} is on the {info['tier']} tier with a {pct}% discount."
    return "Company not found."


TOOL_IMPL = {
    "get_product_price": get_product_price,
    "get_customer_discount": get_customer_discount,
}

tools = [
    {
        "name": "get_product_price",
        "description": "Get the monthly price of a product plan.",
        "parameters": {
            "type": "object",
            "properties": {
                "product_name": {
                    "type": "string",
                    "description": "Plan name (starter, professional, enterprise)",
                }
            },
            "required": ["product_name"],
        },
    },
    {
        "name": "get_customer_discount",
        "description": "Get the discount and tier for a customer company.",
        "parameters": {
            "type": "object",
            "properties": {
                "company": {
                    "type": "string",
                    "description": "Company name (acme_corp, startup_inc)",
                }
            },
            "required": ["company"],
        },
    },
]
