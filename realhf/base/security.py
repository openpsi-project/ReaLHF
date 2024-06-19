def read_key(service, name="default"):
    with open(f"/data/marl/keys/{service}/{name}", "r") as f:
        return f.read().strip()
