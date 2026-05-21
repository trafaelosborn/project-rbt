# Old Spanish Validator Packet

Drop attested Old Spanish or early Iberian texts here.

Rules:

- attested texts only
- avoid reconstructed "Vulgar Iberian" bundles
- document date, region, and source for each text

Once texts are present, ingest with:

```powershell
python -m src.ingest.historical --name old_spanish --language spanish --period "Old Spanish"
```
