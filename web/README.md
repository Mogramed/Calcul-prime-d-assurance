# Nova Assurances Web

Frontend Next.js du site client Nova Assurances.

## Commandes utiles

```bash
npm install
npm run codegen
npm run catalog:vehicles
npm run dev
```

Variables principales:

- `API_BASE_URL`: URL serveur de l'API amont pour les routes BFF Next.js.
- `API_AUDIENCE`: audience Cloud Run optionnelle pour obtenir un jeton serveur a serveur.
- `COOKIE_SECURE`: laisser `false` en local HTTP et passer a `true` sur un domaine HTTPS.

Le navigateur n'appelle que les routes same-origin `/api/*` du frontend Next.js. Le cookie
`nova_client_id` est gere cote serveur, en `HttpOnly`, puis relaye vers l'API amont via le header
`X-Client-ID`.

Le codegen OpenAPI s'appuie sur `../scripts/export_openapi.py` pour generer `openapi.json`
puis sur `@hey-api/openapi-ts` pour regenerer le client dans `generated/client/`.

Le catalogue `marque -> modele` est genere depuis `data/train.csv` avec
`../scripts/export_vehicle_catalog.py` et versionne dans `data/vehicle-catalog.json`.
