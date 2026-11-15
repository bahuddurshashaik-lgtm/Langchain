# Stores a GitHub Personal Access Token in Windows Credential Manager for this repo.
# The token is typed into a hidden prompt, so it is never echoed, never written to
# disk in plain text, and never lands in your shell history or the chat transcript.
#
# Usage (from the repo root, normal PowerShell — not ISE):
#     powershell -ExecutionPolicy Bypass -File scripts\seed_pat.ps1

$ErrorActionPreference = 'Stop'

$username = 'bahuddurshashaik-lgtm'
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

if ((git config --local --get credential.helper) -ne 'manager') {
    git config --local credential.helper manager
    Write-Host "Configured repo-local credential.helper = manager"
}

$secure = Read-Host "Paste your GitHub PAT (required scopes: repo)" -AsSecureString
$bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
try {
    $token = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
} finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
}

if ([string]::IsNullOrWhiteSpace($token)) {
    throw 'No token entered - nothing stored.'
}

$payload = "protocol=https`nhost=github.com`nusername=$username`npassword=$token`n`n"
$payload | git credential approve

if ($LASTEXITCODE -eq 0) {
    Write-Host 'Token stored in Windows Credential Manager for https://github.com'
    Write-Host 'Verify with:  git ls-remote origin'
} else {
    throw 'git credential approve failed.'
}
