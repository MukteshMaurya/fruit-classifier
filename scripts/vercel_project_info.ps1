$ErrorActionPreference = "Continue"

$auth = Get-Content "$env:APPDATA\com.vercel.cli\Data\auth.json" -Raw | ConvertFrom-Json
$token = $auth.token
$team = "team_S84nkYLktLmjzwaYL7ac2TOa"

$raw = curl.exe -s -H "Authorization: Bearer $token" "https://api.vercel.com/v9/projects/fruit-classifier?teamId=$team"
$project = $raw | ConvertFrom-Json
Write-Host "name: $($project.name)  id: $($project.id)"
Write-Host "--- protection-related fields ---"
$project.PSObject.Properties | Where-Object {
  $_.Name -match "protection|Protection|deployment|Deployment"
} | ForEach-Object {
  $val = $_.Value | ConvertTo-Json -Compress -Depth 5
  if ($val.Length -gt 300) { $val = $val.Substring(0, 300) + "..." }
  Write-Host "$($_.Name): $val"
}
