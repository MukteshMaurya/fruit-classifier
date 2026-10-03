$ErrorActionPreference = "Stop"

function Get-CredPassword($hostName) {
  $out = "protocol=https`nhost=$hostName`n" | git credential fill 2>$null
  $line = $out | Select-String "^password=" | Select-Object -First 1
  if ($line) { return ($line.Line -replace "^password=", "") }
  return $null
}

$ghToken = Get-CredPassword "github.com"
$authHeader = "token $ghToken"

$existing = curl.exe -s -o NUL -w "%{http_code}" -H "Authorization: $authHeader" https://api.github.com/repos/MukteshMaurya/fruit-classifier
Write-Host "existing repo status: $existing"

if ($existing -eq "404") {
  $bodyFile = [System.IO.Path]::GetTempFileName() + ".json"
  @{
    name = "fruit-classifier"
    description = "Fruit Classification AI web app - FastAPI + ONNX Swin model (113 fruit classes), Vercel frontend"
    private = $false
    has_issues = $true
    has_projects = $false
    has_wiki = $false
  } | ConvertTo-Json -Compress | Set-Content -Path $bodyFile -NoNewline

  $resp = curl.exe -s -X POST -H "Authorization: $authHeader" -H "Content-Type: application/json" --data "@$bodyFile" https://api.github.com/user/repos
  Remove-Item $bodyFile -Force
  $repo = $resp | ConvertFrom-Json
  Write-Host "created repo: $($repo.html_url)"
  Write-Host "clone url: $($repo.clone_url)"
} else {
  Write-Host "repo already exists"
  $repo = (curl.exe -s -H "Authorization: $authHeader" https://api.github.com/repos/MukteshMaurya/fruit-classifier | ConvertFrom-Json)
  Write-Host "repo url: $($repo.html_url)"
}
