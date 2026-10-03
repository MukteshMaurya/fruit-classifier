$ErrorActionPreference = "Continue"

$auth = Get-Content "$env:APPDATA\com.vercel.cli\Data\auth.json" -Raw | ConvertFrom-Json
$token = $auth.token
$team = "team_S84nkYLktLmjzwaYL7ac2TOa"
$projectId = "prj_DAeTbD6CjbCNXqsq1aw0KApiLqHt"

$bodyFile = [System.IO.Path]::GetTempFileName() + ".json"
'{"ssoProtection":null}' | Set-Content -Path $bodyFile -NoNewline

$resp = curl.exe -s -i -X PATCH -H "Authorization: Bearer $token" `
  -H "Content-Type: application/json" `
  --data "@$bodyFile" `
  "https://api.vercel.com/v9/projects/$projectId/?teamId=$team"
Remove-Item $bodyFile -Force
Write-Host "--- PATCH response ---"
Write-Host ($resp | Out-String)
