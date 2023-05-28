from packaging import version
def parse_version(ver: str):
    return version.parse(ver)