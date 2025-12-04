"""
Lesson 9: Custom Tools
"""

import os
import re
from datetime import datetime
from typing import Dict
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama

load_dotenv()

model = os.getenv("model_2")

llm = ChatOllama(
    model=model,
    temperature=0,
    num_ctx=2048,
    num_predict=512
)


def example_1_validation_tools():
    """Example 1: Tools with input validation"""
    print("=" * 50)
    print("Example 1: Validation Tools")
    print("=" * 50)
    
    @tool
    def validate_email(email: str) -> str:
        """
        Validates an email address format.
        Input: email address (e.g., 'user@example.com')
        Returns: validation result
        """
        try:
            # Basic email validation
            pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if re.match(pattern, email):
                return f"✅ '{email}' is a VALID email address"
            else:
                return f"❌ '{email}' is INVALID email format"
        except Exception as e:
            return f"Error validating email: {str(e)}"
    
    @tool
    def validate_phone(phone: str) -> str:
        """
        Validates a phone number (US format).
        Input: phone number (e.g., '123-456-7890' or '1234567890')
        Returns: validation result
        """
        try:
            # Remove common separators
            clean_phone = re.sub(r'[- ().]', '', phone)
            
            if len(clean_phone) == 10 and clean_phone.isdigit():
                formatted = f"({clean_phone[:3]}) {clean_phone[3:6]}-{clean_phone[6:]}"
                return f"✅ Valid phone: {formatted}"
            else:
                return f"❌ Invalid phone number format"
        except Exception as e:
            return f"Error validating phone: {str(e)}"
    
    @tool
    def validate_url(url: str) -> str:
        """
        Validates a URL format.
        Input: URL (e.g., 'https://example.com')
        Returns: validation result
        """
        try:
            pattern = r'^https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}.*$'
            if re.match(pattern, url):
                return f"✅ '{url}' is a VALID URL"
            else:
                return f"❌ '{url}' is INVALID URL format"
        except Exception as e:
            return f"Error validating URL: {str(e)}"
    
    tools = [validate_email, validate_phone, validate_url]
    
    agent = create_react_agent(llm, tools)
    
    prompt = input("Check if the email entered is valid or not : ")    
    print("\n🤖 Task: Validate contact information")
    result = agent.invoke({
        "messages": [("system", """
                      You are a validation assistant.
                      Validate the following email information using the appropriate tool.
                      """),
                     ("human", prompt)
                     ]
    })
    print(f"\n✅ Answer: {result['messages'][-1].content}\n")


def example_2_data_processing_tools():
    """Example 2: Tools for data processing and analysis"""
    print("=" * 50)
    print("Example 2: Data Processing Tools")
    print("=" * 50)
    
    @tool
    def parse_json(json_str: str) -> str:
        """
        Parses and validates JSON string.
        Input: JSON string
        Returns: parsed data or error
        """
        try:
            data = json.loads(json_str)
            return f"Valid JSON with {len(data)} items: {json.dumps(data, indent=2)}"
        except json.JSONDecodeError as e:
            return f"Invalid JSON: {str(e)}"
    
    @tool
    def calculate_statistics(numbers_str: str) -> str:
        """
        Calculates statistics for a list of numbers.
        Input: comma-separated numbers (e.g., '1,2,3,4,5')
        Returns: mean, median, min, max, sum
        """
        try:
            numbers = [float(n.strip()) for n in numbers_str.split(',')]
            if not numbers:
                return "Error: No numbers provided"
            
            mean = sum(numbers) / len(numbers)
            sorted_nums = sorted(numbers)
            median = sorted_nums[len(numbers)//2] if len(numbers) % 2 else \
                    (sorted_nums[len(numbers)//2-1] + sorted_nums[len(numbers)//2]) / 2
            
            return f"""Statistics:
- Count: {len(numbers)}
- Sum: {sum(numbers)}
- Mean: {mean:.2f}
- Median: {median}
- Min: {min(numbers)}
- Max: {max(numbers)}"""
        except Exception as e:
            return f"Error calculating statistics: {str(e)}"
    
    @tool
    def format_data(data_str: str) -> str:
        """
        Formats data into a readable table format.
        Input: data in format 'key1:value1,key2:value2'
        Returns: formatted table
        """
        try:
            items = data_str.split(',')
            formatted = "| Key | Value |\n|-----|-------|\n"
            for item in items:
                if ':' in item:
                    key, value = item.split(':', 1)
                    formatted += f"| {key.strip()} | {value.strip()} |\n"
            return formatted
        except Exception as e:
            return f"Error formatting data: {str(e)}"
    
    tools = [parse_json, calculate_statistics, format_data]
    
    agent = create_react_agent(llm, tools)
    
    print("\n🤖 Task: Calculate statistics for test scores")
    result = agent.invoke({
        "messages": [("user", "Calculate statistics for these test scores: 85, 92, 78, 95, 88, 90")]
    })
    print(f"\n✅ Answer: {result['messages'][-1].content}\n")


def example_3_datetime_tools():
    """Example 3: Tools for date and time operations"""
    print("=" * 50)
    print("Example 3: DateTime Tools")
    print("=" * 50)
    
    @tool
    def get_current_datetime() -> str:
        """Returns the current date and time."""
        now = datetime.now()
        return now.strftime("%Y-%m-%d %H:%M:%S")
    
    @tool
    def calculate_age(birth_year: str) -> str:
        """
        Calculates age from birth year.
        Input: birth year (e.g., '1990')
        Returns: current age
        """
        try:
            year = int(birth_year)
            current_year = datetime.now().year
            age = current_year - year
            return f"Age: {age} years old"
        except ValueError:
            return "Error: Invalid year format"
    
    @tool
    def days_until_date(date_str: str) -> str:
        """
        Calculates days until a future date.
        Input: date in format 'YYYY-MM-DD'
        Returns: number of days
        """
        try:
            target_date = datetime.strptime(date_str, "%Y-%m-%d")
            today = datetime.now()
            delta = target_date - today
            days = delta.days
            
            if days > 0:
                return f"{days} days until {date_str}"
            elif days == 0:
                return f"Today is {date_str}!"
            else:
                return f"{date_str} was {abs(days)} days ago"
        except ValueError:
            return "Error: Invalid date format. Use YYYY-MM-DD"
    
    @tool
    def day_of_week(date_str: str) -> str:
        """
        Gets the day of week for a date.
        Input: date in format 'YYYY-MM-DD'
        Returns: day name
        """
        try:
            date = datetime.strptime(date_str, "%Y-%m-%d")
            day_name = date.strftime("%A")
            return f"{date_str} is a {day_name}"
        except ValueError:
            return "Error: Invalid date format. Use YYYY-MM-DD"
    
    tools = [get_current_datetime, calculate_age, days_until_date, day_of_week]
    
    agent = create_react_agent(llm, tools)
    
    print("\n🤖 Task: Date calculations")
    result = agent.invoke({
        "messages": [("user", "What's today's date and what day of the week will 2025-12-25 be?")]
    })
    print(f"\n✅ Answer: {result['messages'][-1].content}\n")


def example_4_business_logic_tools():
    """Example 4: Tools with business logic"""
    print("=" * 50)
    print("Example 4: Business Logic Tools")
    print("=" * 50)
    
    @tool
    def calculate_discount(input_str: str) -> str:
        """
        Calculates discounted price.
        Input: 'price,discount_percent' (e.g., '100,20' for $100 with 20% off)
        Returns: final price after discount
        """
        try:
            parts = input_str.split(',')
            price = float(parts[0])
            discount_percent = float(parts[1])
            
            if price < 0 or discount_percent < 0 or discount_percent > 100:
                return "Error: Invalid values"
            
            discount_amount = price * (discount_percent / 100)
            final_price = price - discount_amount
            
            return f"""Price breakdown:
- Original: ${price:.2f}
- Discount: {discount_percent}% (${discount_amount:.2f})
- Final price: ${final_price:.2f}"""
        except Exception as e:
            return f"Error: {str(e)}"
    
    @tool
    def calculate_tax(input_str: str) -> str:
        """
        Calculates tax on an amount.
        Input: 'amount,tax_rate' (e.g., '100,8.5' for $100 with 8.5% tax)
        Returns: amount with tax
        """
        try:
            parts = input_str.split(',')
            amount = float(parts[0])
            tax_rate = float(parts[1])
            
            tax_amount = amount * (tax_rate / 100)
            total = amount + tax_amount
            
            return f"""Tax calculation:
- Subtotal: ${amount:.2f}
- Tax ({tax_rate}%): ${tax_amount:.2f}
- Total: ${total:.2f}"""
        except Exception as e:
            return f"Error: {str(e)}"
    
    @tool
    def calculate_tip(input_str: str) -> str:
        """
        Calculates tip for a bill.
        Input: 'bill_amount,tip_percent' (e.g., '50,20' for $50 with 20% tip)
        Returns: tip amount and total
        """
        try:
            parts = input_str.split(',')
            bill = float(parts[0])
            tip_percent = float(parts[1])
            
            tip_amount = bill * (tip_percent / 100)
            total = bill + tip_amount
            
            return f"""Tip calculation:
- Bill: ${bill:.2f}
- Tip ({tip_percent}%): ${tip_amount:.2f}
- Total: ${total:.2f}"""
        except Exception as e:
            return f"Error: {str(e)}"
    
    tools = [calculate_discount, calculate_tax, calculate_tip]
    
    agent = create_react_agent(llm, tools)
    
    print("\n🤖 Task: Shopping calculation")
    result = agent.invoke({
        "messages": [("user", "I have an item that costs $80 with a 15% discount. "
                 "Then I need to add 8% tax. What's the final price?")]
    })
    print(f"\n✅ Answer: {result['messages'][-1].content}\n")


def example_5_structured_tool():
    """Example 5: Using StructuredTool with multiple parameters"""
    print("=" * 50)
    print("Example 5: Structured Tool with Multiple Parameters")
    print("=" * 50)
    
    class EmailInput(BaseModel):
        recipient: str = Field(description="Email recipient address")
        subject: str = Field(description="Email subject line")
        body: str = Field(description="Email body text")
    
    def send_email(recipient: str, subject: str, body: str) -> str:
        """
        Simulates sending an email.
        """
        # In real implementation, this would send actual email
        return f"""Email sent successfully!
To: {recipient}
Subject: {subject}
Body: {body[:50]}..."""
    
    email_tool = StructuredTool.from_function(
        func=send_email,
        name="SendEmail",
        description="Sends an email with recipient, subject, and body",
        args_schema=EmailInput
    )
    
    tools = [email_tool]
    
    agent = create_react_agent(llm, tools)
    
    print("\n🤖 Task: Send an email")
    result = agent.invoke({
        "messages": [("user", "Send an email to john@example.com with subject 'Meeting Tomorrow' "
                 "and body 'Let's meet at 10 AM to discuss the project.'")]
    })
    print(f"\n✅ Answer: {result['messages'][-1].content}\n")


def main():
    """Run all examples"""
    print("\n🚀 LangChain Custom Tools Examples\n")
    
    if not os.getenv("model_2"):
        print("❌ Error: model not found!")
        return
    
    try:
        example_1_validation_tools()
        # example_2_data_processing_tools()
        # example_3_datetime_tools()
        # example_4_business_logic_tools()
        # example_5_structured_tool()
        
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
