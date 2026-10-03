$ErrorActionPreference = "Continue"

$auth = Get-Content "$env:APPDATA\com.vercel.cli\Data\auth.json" -Raw | ConvertFrom-Json
$token = $auth.token
$team = "team_S84nkYLktLmjzwaYL7ac2TOa"

$deployments = curl.exe -s -H "Authorization: Bearer $token" "https://api.vercel.com/v6/deployments?teamId=$team&projectId=prj_DAeTbD6CjbCNXqsq1aw0KApiLqHt&limit=3" | ConvertFrom-Json
foreach ($d in $deployments.deployments) {
  $url = ($d.url)
  $state = $d.state
  $created = $d.createdAt
  Write-Host "deployment: $state  url: $url  created: $created"
}
