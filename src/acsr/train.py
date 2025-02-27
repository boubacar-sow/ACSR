import subprocess
def text_to_ipa(text, language="fr"):
    """
    Convert text to IPA using espeak-ng.
    """
    # Remove special characters
    #text = text.replace("?", "").replace("!", "").replace(".", "").replace(",", "").replace(":", "").replace(";", "").replace("'", "").replace("-", " ")

    command = ["espeak-ng", "-v", language, "-q", "--ipa"]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate(input=text.encode())
    ipa_output = stdout.decode().strip()
    ipa_output = ipa_output.replace("ˈ", "").replace("ˌ", "").replace("-", "").replace("\n", " ").replace("(en)", "").replace("(fr)", "")

    return ipa_output

print(text_to_ipa(' "Le malade : Le féminin de ""le malade"" est ""la malade""."'))