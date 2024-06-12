import json

def convert_scheme_structure(schemes):
    formatted_schemes = []
    for scheme in schemes:
        tag = "Scheme" + scheme["sr_no"]
        patterns = [scheme["scheme_name"]]
        #patterns = [pattern.strip() for pattern in patterns]
        responses = [scheme["details"]]
        
        formatted_schemes.append({
            "tag": tag,
            "patterns": patterns,
            "responses": responses
        })
    
    return formatted_schemes

# Read the existing file structure
with open("C:\\Users\\91875\\Desktop\\SchemeBot\\AI-Chatbot\\myschemescrape.json", "r") as file:
    schemes = json.load(file)

# Convert the structure to the desired format
formatted_schemes = convert_scheme_structure(schemes)

# Write the formatted data to a new JSON file
with open("C:\\Users\\91875\\Desktop\\SchemeBot\\AI-Chatbot\\formatted_structure.json", "w") as file:
    json.dump(formatted_schemes, file, indent=4)
