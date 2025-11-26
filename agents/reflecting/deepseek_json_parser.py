#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import json
import re
from typing import Optional, Dict, Any, List

class DeepSeekJSONParser:
    """
    Utility plugin for extracting JSON configurations from DeepSeek RL responses
    that contain thinking processes, HTML details tags, and mixed content.
    """
    
    def __init__(self, debug: bool = True):
        self.debug = debug
    
    def _log(self, message: str):
        """Log debug messages if debug mode is enabled"""
        if self.debug:
            print(message)
    
    def extract_json(self, response: str, required_fields: Optional[List[str]] = None) -> Optional[Dict[Any, Any]]:
        """
        Extract JSON from DeepSeek RL response using multiple strategies.
        
        Args:
            response: Raw response text from DeepSeek RL
            required_fields: List of required fields that must be present in the JSON
            
        Returns:
            Parsed JSON dictionary or None if extraction fails
        """
        if not isinstance(response, str):
            return response if isinstance(response, dict) else None
        
        self._log("🔍 Parsing DeepSeek response for JSON configuration...")
        
        json_str = None
        
        # Strategy 1: Look for markdown code blocks
        if "```json" in response:
            self._log("   - Strategy 1: Found markdown code block")
            start = response.find("```json") + 7
            end = response.find("```", start)
            json_str = response[start:end].strip()
        
        # Strategy 2: Use proper JSON parsing to find complete objects
        else:
            self._log("   - Strategy 2: Searching for complete JSON objects")
            json_str = self._extract_complete_json_objects(response, required_fields)
        
        # Clean and validate the JSON string
        if json_str:
            return self._clean_and_parse_json(json_str, required_fields)
        else:
            self._log("   - ❌ No valid JSON found")
            return None
    
    def _extract_complete_json_objects(self, text: str, required_fields: Optional[List[str]] = None) -> Optional[str]:
        """Extract complete JSON objects using bracket counting and field matching"""
        
        # Method 1: Look for JSON objects with specific required fields
        if required_fields:
            for field in required_fields:
                pattern = rf'\{{\s*"{field}"[^}}]*'
                matches = [m.start() for m in re.finditer(pattern, text)]
                
                if matches:
                    self._log(f"   - Found {len(matches)} potential JSON start positions with '{field}'")
                    
                    for i, start_pos in enumerate(matches):
                        candidate = self._extract_complete_json_from_position(text, start_pos)
                        if candidate:
                            # Test if this JSON contains all required fields
                            try:
                                test_data = json.loads(candidate)
                                if all(field in test_data for field in required_fields):
                                    self._log(f"   - ✅ Found valid JSON with required fields in candidate {i+1}")
                                    return candidate
                            except json.JSONDecodeError:
                                continue
        
        # Method 2: Fallback to pattern matching
        self._log("   - Fallback: using regex pattern matching")
        json_matches = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        
        if json_matches:
            # If we have required fields, find JSON containing them
            if required_fields:
                for candidate in json_matches:
                    if all(f'"{field}"' in candidate for field in required_fields):
                        return candidate
            
            # Otherwise return the largest JSON object
            return max(json_matches, key=len)
        
        return None
    
    def _extract_complete_json_from_position(self, text: str, start_pos: int) -> Optional[str]:
        """Extract complete JSON object using bracket counting from a given position"""
        remaining_text = text[start_pos:]
        
        bracket_count = 0
        in_string = False
        escape_next = False
        end_pos = 0
        
        for j, char in enumerate(remaining_text):
            if escape_next:
                escape_next = False
                continue
                
            if char == '\\':
                escape_next = True
                continue
                
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
                
            if not in_string:
                if char == '{':
                    bracket_count += 1
                elif char == '}':
                    bracket_count -= 1
                    if bracket_count == 0:
                        end_pos = j + 1
                        break
        
        if end_pos > 0:
            candidate_json = remaining_text[:end_pos]
            self._log(f"   - Extracted candidate: {len(candidate_json)} chars")
            return candidate_json
        
        return None
    
    def _clean_and_parse_json(self, json_str: str, required_fields: Optional[List[str]] = None) -> Optional[Dict[Any, Any]]:
        """Clean and parse JSON string with validation"""
        self._log(f"   - Selected JSON string: {len(json_str)} chars")
        
        # Clean the JSON string
        json_str = json_str.strip()
        
        # Handle potential HTML entities
        json_str = json_str.replace('&quot;', '"').replace('&gt;', '>').replace('&lt;', '<')
        
        # Remove any trailing text after the JSON (keep only first complete object)
        if '\n' in json_str and not json_str.strip().endswith('}'):
            json_str = json_str.split('\n')[0] if '\n' in json_str else json_str
        
        self._log(f"   - Preview: {json_str[:200]}...")
        
        try:
            config_data = json.loads(json_str)
            self._log("   - ✅ JSON parsed successfully")
            
            # Validate required fields if specified
            if required_fields:
                missing_fields = [field for field in required_fields if field not in config_data]
                if missing_fields:
                    self._log(f"   - ❌ Missing required fields: {missing_fields}")
                    return None
                self._log("   - ✅ Required fields validated")
            
            return config_data
            
        except json.JSONDecodeError as e:
            self._log(f"   - ❌ JSON parsing failed: {e}")
            return None
    
    def extract_and_validate(self, response: str, required_fields: List[str]) -> Optional[Dict[Any, Any]]:
        """
        Extract JSON and validate it contains all required fields.
        Convenience method that combines extraction and validation.
        """
        return self.extract_json(response, required_fields)
    
    def debug_response(self, response: str) -> None:
        """Debug helper to analyze response structure"""
        print("🔍 DeepSeek Response Analysis:")
        print(f"   - Response length: {len(response)} chars")
        print(f"   - Contains markdown: {'```json' in response}")
        print(f"   - Contains HTML details: {'<details' in response}")
        
        # Look for JSON-like patterns
        json_patterns = re.findall(r'\{[^}]*\}', response)
        print(f"   - Simple JSON patterns found: {len(json_patterns)}")
        
        # Show segments containing common JSON field names
        common_fields = ['segment_per_ring', 'segment_order', 'prompt_points', 'resolution']
        for field in common_fields:
            if field in response:
                start = max(0, response.find(field) - 50)
                end = min(len(response), response.find(field) + 100)
                print(f"   - Context around '{field}': ...{response[start:end]}...")
                break


# Convenience functions for backward compatibility
def extract_json_from_deepseek(response: str, required_fields: Optional[List[str]] = None, debug: bool = True) -> Optional[Dict[Any, Any]]:
    """Convenience function to extract JSON from DeepSeek response"""
    parser = DeepSeekJSONParser(debug=debug)
    return parser.extract_json(response, required_fields)

def debug_deepseek_response(response: str) -> None:
    """Convenience function to debug DeepSeek response structure"""
    parser = DeepSeekJSONParser(debug=True)
    parser.debug_response(response) 