# Middle French Validator Packet

Drop attested Middle French texts here.

Rules:

- attested texts only
- keep witnesses / editions listed below as they are added
- do not mix Old French or modern French material into this folder

Suggested metadata to record:

- title
- witness or edition
- approximate date range
- region
- source URL or archive reference

Once texts are present, ingest with:

```powershell
python -m src.ingest.historical --name middle_french --language french --period "Middle French"
```
