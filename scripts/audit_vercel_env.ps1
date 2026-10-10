# Vercel project env var audit (read-only).
# Reads the LOCAL Vercel CLI auth token from the standard CLI
# location; the token itself is never printed or transmitted
# anywhere except api.vercel.com over HTTPS.
$ErrorActionPreference = "Continue"

$authPath = "$env:APPDATA\com.vercel.cli\Data\auth.json"
if (-not (Test-Path $authPath)) {
  Write-Host "NO_VERCEL_AUTH"
  return
}
$auth = Get-Content $authPath -Raw | ConvertFrom-Json
$token = $auth.token
$team = "team_S84nkYLktLmjzwaYL7ac2TOa"
$projectId = "prj_DAeTbD6CjbCNXqsq1aw0KApiLqHt"

$resp = curl.exe -s -H "Authorization: Bearer $token" `
  "https://api.vercel.com/v10/projects/$projectId/env?teamId=$team"
$data = $resp | ConvertFrom-Json

if ($data.error) {
  Write-Host "API_ERROR: $($data.error.message)"
  return
}

Write-Host "Environment variables on Vercel project 'fruit-classifier':"
foreach ($envVar in $data.envs) {
  $val = ""
  if ($envVar.key -like "VITE_*") {
    # Frontend-exposed vars are public by design; show them so we
    # can confirm they hold only the backend URL, not a secret.
    $val = " => $($envVar.value)"
  }
  Write-Host ("  {0}  target={1} type={2}{3}" -f $envVar.key, ($envVar.target -join ","), $envVar.type, $val)
}
Write-Host "Total: $($data.envs.Count)"
