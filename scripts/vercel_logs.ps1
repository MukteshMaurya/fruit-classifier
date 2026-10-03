$ErrorActionPreference = "Continue"

$auth = Get-Content "$env:APPDATA\com.vercel.cli\Data\auth.json" -Raw | ConvertFrom-Json
$token = $auth.token
$team = "team_S84nkYLktLmjzwaYL7ac2TOa"

$deps = curl.exe -s -H "Authorization: Bearer $token" "https://api.vercel.com/v6/deployments?teamId=$team&projectId=prj_DAeTbD6CjbCNXqsq1aw0KApiLqHt&limit=1" | ConvertFrom-Json
$d = $deps.deployments[0]
Write-Host "deployment id: $($d.id)  state: $($d.state)  url: $($d.url)"
Write-Host "--- meta ---"
$d.meta | ConvertTo-Json -Depth 4
Write-Host "--- error ---"
$d.error | ConvertTo-Json -Depth 4

$logs = curl.exe -s -H "Authorization: Bearer $token" "https://api.vercel.com/v2/deployments/$($d.id)/events?teamId=$team&limit=40" | ConvertFrom-Json
Write-Host "--- build log (last events) ---"
foreach ($e in $logs.events) {
  $msg = $e.message
  if ($msg) {
    $line = ($msg | ConvertTo-Json -Compress)
    if ($line.Length -gt 240) { $line = $line.Substring(0, 240) + "..." }
    Write-Host $line
  }
}
