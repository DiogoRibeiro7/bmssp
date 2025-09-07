# Exporting Results

## JSON
```bash
ssspx --edges graph.mtx --export-json out.json
```

## GraphML
```bash
ssspx --edges graph.jsonl --export-graphml out.graphml

Use ``--format`` to override if the input lacks an extension:

```bash
ssspx --edges graph --format csv --export-json out.json
```
```
