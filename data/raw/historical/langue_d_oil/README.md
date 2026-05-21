# Langue d'oil Validator Packet

Drop attested non-Anglo-Norman langue d'oil texts here.

Rules:

- attested texts only
- keep specific varieties documented in this README as material is added
- do not merge this folder with Old French or Middle French by default

Recommended note fields:

- variety name
- date
- region
- source

Once texts are present, ingest with:

```powershell
python -m src.ingest.historical --name langue_d_oil --language french --period "Langue d'oil"
```
