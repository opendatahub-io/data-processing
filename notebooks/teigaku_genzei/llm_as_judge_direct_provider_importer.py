import os
import json
import logging
import catalog_util


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    system_catalog_path = catalog_util.get_system_catalog_path()
    if system_catalog_path is None:
        logging.error("No system catalog")
        exit(1)
    local_catalog_path = catalog_util.LOCAL_CATALOG_PATH
    base_source_path = os.path.join(system_catalog_path, "metrics/llm_as_judge/direct")
    base_target_path = os.path.join(local_catalog_path, "metrics/llm_as_judge/direct_positional_bias")
    exclude_set = {"criteria"}

    logging.info(f"Copying artifacts from {base_source_path} to {base_target_path}")
    for dir in os.listdir(base_source_path):
        if dir in exclude_set:
            continue
        logging.info(f"Processing {dir}")
        provider_source_path = os.path.join(base_source_path, dir)
        provider_target_path = os.path.join(base_target_path, dir)
        if not os.path.exists(provider_target_path):
            os.makedirs(provider_target_path)
        for file in os.listdir(provider_source_path):
            if file.endswith(".json"):
                logging.info(f"Processing {file}")
                source_path = os.path.join(provider_source_path, file)
                target_path = os.path.join(provider_target_path, file)
                with open(source_path, "r", encoding="utf-8") as f:
                    source_artifact = json.load(f)
                    source_artifact["__type__"] = "llm_judge_direct_positional_bias"
                    with open(target_path, "w", encoding="utf-8") as wf:
                        json.dump(source_artifact, wf, indent=4)
                        logging.info(f"Wrote {target_path}")


