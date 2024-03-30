import pydantic


# класс(-ы), описывающий выход сервиса
class ServiceOutput(pydantic.BaseModel):
    """_summary_

    Args:
        pydantic (_type_): _description_

    Returns:
        _type_: _description_
    """

    objects: list
