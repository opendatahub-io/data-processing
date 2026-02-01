from unitxt import register_local_catalog
from unitxt.catalog import get_from_catalog, get_local_catalogs_paths
from unitxt.artifact import UnitxtArtifactNotFoundError

LOCAL_CATALOG_PATH="local_catalog"

register_local_catalog(LOCAL_CATALOG_PATH)

def is_artifact_in_catalog(artifact_name: str, catalog_path: str=LOCAL_CATALOG_PATH) -> bool:
    try:
        _ = get_from_catalog(artifact_name, catalog_path=catalog_path)
    except UnitxtArtifactNotFoundError:
        return False
    return True

def get_system_catalog_path() -> str | None:
    paths = get_local_catalogs_paths()
    print(paths)

    return paths[-1] if len(paths) > 0 else None

get_system_catalog_path()
