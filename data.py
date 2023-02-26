from pydantic import BaseModel


class Product(BaseModel):
    title: str
    description: str
    variant_sku: list[str]
    handle: str
    thumbnail: str


class SearchResponse(BaseModel):
    hits: list[Product] = [
        Product(
            title="Medusa T-Shirt",
            description="Reimagine the feeling of a classic T-shirt. With our cotton T-shirts, everyday essentials no longer have to be ordinary.",
            variant_sku=[],
            handle="t-shirt",
            thumbnail="https://medusa-public-images.s3.eu-west-1.amazonaws.com/tee-black-front.png",
        ),
        Product(
            title="Nike Unisex Feather lite Blue Caps",
            description="Royal blue featherlight tennis cap made of polyester and recycled content, with small mesh panels on the crown, nike Swoosh embroidered on the front of the crown.",
            variant_sku=[],
            handle="cap",
            thumbnail="http://assets.myntassets.com/v1/images/style/properties/7c92ff84d5e91203ba60b622e0431dec_images.jpg",
        ),
    ]
