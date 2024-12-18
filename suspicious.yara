// SuspiciousFile.yara
rule MalwareSample
{
    meta:
        description = "Detect a specific malware sample"
        author = "Your Name"
    strings:
        $malicious_string = "malware_sample"
    condition:
        $malicious_string
}

rule SuspiciousExecutable
{
    meta:
        description = "Detect suspicious executable files"
    strings:
        $mz_header = { 4D 5A }  // MZ header
        $exe_string = ".exe"
    condition:
        $mz_header or $exe_string
}

rule CreditCardNumbers
{
    meta:
        description = "Detect potential credit card numbers in files"
    strings:
        $visa = /\b4[0-9]{12}(?:[0-9]{3})?\b/
        $mastercard = /\b5[1-5][0-9]{14}\b/
        $amex = /\b3[47][0-9]{13}\b/
    condition:
        $visa or $mastercard or $amex
}

rule PHPWebShell
{
    meta:
        description = "Detect potential PHP web shells"
    strings:
        $eval_base64 = /eval\(base64_decode/
        $cmd_exec = /exec\(\$_POST/
    condition:
        $eval_base64 or $cmd_exec
}

rule HiddenPEFile
{
    meta:
        description = "Detect hidden PE files embedded in documents"
    strings:
        $pe_header = { 50 45 00 00 }  // PE\0\0
        $mz_header = { 4D 5A }        // MZ header
    condition:
        $pe_header or $mz_header
}

rule EmailAddresses
{
    meta:
        description = "Detect email addresses in text"
    strings:
        $email_pattern = /[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/
    condition:
        $email_pattern
}

rule ObfuscatedJavaScript
{
    meta:
        description = "Detect obfuscated JavaScript code"
    strings:
        $eval = "eval("
        $escape = "unescape("
        $hex_string = /\\x[a-f0-9]{2}/
    condition:
        $eval or $escape or $hex_string
}

rule RansomwareDetection
{
    meta:
        description = "Detect ransomware strings"
    strings:
        $ransom_note = "Your files have been encrypted"
        $payment = "Send Bitcoin to the following address"
    condition:
        $ransom_note or $payment
}

rule EmbeddedZipFile
{
    meta:
        description = "Detect embedded ZIP files in documents"
    strings:
        $zip_header = { 50 4B 03 04 }  // PK.. (ZIP signature)
    condition:
        $zip_header
}

rule SuspiciousURLs
{
    meta:
        description = "Detect suspicious URLs in files"
    strings:
        $http = "http://"
        $https = "https://"
        $ip_url = /http[s]?:\/\/[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+/
    condition:
        $http or $https or $ip_url
}

rule SocialSecurityNumbers
{
    meta:
        description = "Detect potential SSNs in text"
    strings:
        $ssn = /\b\d{3}-\d{2}-\d{4}\b/
    condition:
        $ssn
}

rule CommonPasswords
{
    meta:
        description = "Detect commonly used passwords"
    strings:
        $password1 = "123456"
        $password2 = "password"
        $password3 = "qwerty"
    condition:
        any of ($password*)
}

rule Base64Strings
{
    meta:
        description = "Detect base64 encoded strings"
    strings:
        $base64_pattern = /[A-Za-z0-9+/]{40,}={0,2}/
    condition:
        $base64_pattern
}
