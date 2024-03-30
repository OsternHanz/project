import pydantic


class ServiceConfig(pydantic.BaseModel):
    """_summary_

    Args:
        pydantic (_type_): _description_

    Returns:
        _type_: _description_
    """

    target_width: int
    target_height: int
