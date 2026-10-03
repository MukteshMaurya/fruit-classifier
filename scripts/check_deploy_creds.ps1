$ErrorActionPreference = "Stop"

function Get-CredPassword($hostName) {
  $out = "protocol=https`nhost=$hostName`n" | git credential fill 2>$null
  $line = $out | Select-String "^password=" | Select-Object -First 1
  if ($line) { return ($line.Line -replace "^password=", "") }
  return $null
}

# --- GitHub ---
$ghToken = Get-CredPassword "github.com"
if ($ghToken) {
  $headers = @{ Authorization = "token $ghToken"; Accept = "application/vnd.github+json" }
  try {
    $me = curl.exe -s -H $headers.Authorization -H "Accept: application/vnd.github+json" https://api.github.com/user | ConvertFrom-Json
    Write-Host "GITHUB_OK user=$($me.login) name=$($me.name)"
    $rate = curl.exe -s -H $headers.Authorization https://api.github.com/rate_limit | ConvertFrom-Json
    Write-Host "GITHUB_RATE remaining=$($rate.rate.remaining)"
  } catch {
    Write-Host "GITHUB_FAIL $_"
  }
} else {
  Write-Host "GITHUB_NO_CRED"
}

# --- Hugging Face ---
$hfPass = Get-CredPassword "huggingface.co"
if ($hfPass) {
  try {
    $who = curl.exe -s -H "Authorization: Bearer $hfPass" https://huggingface.co/api/whoami-v2 | ConvertFrom-Json
    if ($who.name) {
      Write-Host "HF_OK user=$($who.name) type=$($who.type)"
    } else {
      Write-Host "HF_FAIL response=$($who | ConvertTo-Json -Compress -Depth 3)"
    }
  } catch {
    Write-Host "HF_FAIL $_"
  }
} else {
  Write-Host "HF_NO_CRED"
}
