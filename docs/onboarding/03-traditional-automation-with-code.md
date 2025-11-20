# Traditional automation with code

Even with great tools like Selenium and Appium, writing E2E automation is complex and expensive. This doc shows what traditional automation looks like.

## How traditional automation works: Locators

Traditional automation relies on **locators** — identifiers that tell the tool which UI element to interact with.

### Locators on Web

Web pages are built with HTML. Every element has properties you can target:

```html
<button id="login-btn" class="primary">Login</button>
<input type="text" name="username" placeholder="Enter your email">
```

**Common web locators**:
- **ID**: `id=login-btn` (best when available)
- **CSS selector**: `.primary` or `button.primary`
- **XPath**: `//button[text()='Login']` (powerful but fragile)
- **Text**: `text=Login`
- **Name**: `name=username`

### Locators on Mobile

Mobile apps have a view hierarchy. Each element has properties:

```xml
<Button 
  resource-id="com.example:id/login" 
  text="Login" 
  content-desc="Login button"/>
<TextField 
  resource-id="com.example:id/username" 
  hint="Enter username"/>
```

**Common mobile locators**:
- **Resource ID**: `id=com.example:id/login` (Android)
- **Accessibility ID**: `accessibility id=Login button` (iOS/Android)
- **XPath**: `//android.widget.Button[@text='Login']`
- **Text**: `text=Login`

### Key difference: Web vs Mobile

**Web**: More stable IDs and CSS selectors. Developers often add meaningful IDs.

**Mobile**: IDs are less common. Many elements lack `resource-id` or `content-desc`, forcing fragile XPath locators based on position.

## Raw automation code example

Here's a very basic Selenium test in Python:

```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

def test_user_login():
    # Setup driver
    driver = webdriver.Chrome()
    driver.get("https://example.com/login")
    
    try:
        # Wait for username field
        username_field = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "username"))
        )
        username_field.send_keys("alice@example.com")
        
        # Find and fill password
        password_field = driver.find_element(By.ID, "password")
        password_field.send_keys("secret123")
        
        # Click login button
        login_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[@type='submit']"))
        )
        login_button.click()
        
        # Wait and verify success
        time.sleep(2)
        welcome_text = driver.find_element(By.CSS_SELECTOR, ".welcome-message")
        assert "Welcome" in welcome_text.text
        
    finally:
        driver.quit()
```

## Problems with raw automation code

**Immediate issues**:
- **Verbose**: 30+ lines for a simple login test
- **Technical**: Requires programming knowledge (imports, waits, exception handling)
- **Fragile**: If `id="username"` changes to `id="email"`, test breaks
- **Hard to maintain**: Changing one locator means editing code
- **Not readable**: Non-technical QA can't understand or contribute

### The real problem: scaling to 100+ tests

This example is **basic**. In reality, when you have 100+ tests, the complexity explodes:

- **Design patterns needed**: Page Object Model, Factory patterns, Builder patterns to organize code
- **Strong programming skills required**: Object-oriented programming, inheritance, abstraction
- **Infrastructure code**: Test data management, configuration handling, reporting hooks, retry mechanisms
- **Code review overhead**: Every test change requires reviewing complex Python/Java code
- **High entry barrier**: Junior QA engineers can't contribute without solid programming background

**Result**: E2E automation becomes a software engineering project itself. You need senior developers just to write and maintain tests.

## Core problems with locator-based automation

These problems affect all traditional automation, whether raw code or frameworks:

### 1. Brittle locators

Developer changes `id=submit-button` to `id=submit-btn`. Test breaks.

**Cost**: Manually find and update all affected tests.

### 2. Dynamic content

Dashboard shows "5 new messages" but the number changes based on data.

**Cost**: Tests become data-dependent. Valid builds fail.

### 3. Timing and async loading

Button appears after an API call, but test clicks too early.

**Cost**: Add hardcoded waits (`sleep(3)`), making tests slow and still flaky.

### 4. Environment-specific data

Production shows "John Doe", staging shows "Test User".

**Cost**: Separate tests per environment or complex variable management.

### 5. Cross-platform locator differences

Same button, different locators per platform:
- Web: `id=menu`
- Android: `resource-id=com.app:id/menu`
- iOS: `accessibility id=Menu`

**Cost**: Write and maintain separate tests for each platform.

### 6. Visual elements can't be validated

App shows a map, chart, or image. Locators can't verify content.

**Cost**: Visual bugs slip through. Manual testing still needed.

### 7. Accessibility issues on mobile

Button has no `content-desc` or `resource-id`. Only way to find it is fragile XPath based on position.

**Cost**: Tests break on minor UI rearrangements.

### 8. Complex debugging

Test fails at step 15 of a 20-step flow. You only get "Element not found".

**Cost**: Rerun the flow repeatedly, adding print statements, until you find the issue.

## Why this matters

Each problem seems small, but they compound:

- A test suite with 100 tests
- Each test has ~10 locators
- UI changes once per sprint
- = 1000 locators to potentially maintain
- = Hours of weekly maintenance

**Result**: Teams write fewer E2E tests or let them rot.

## What's next

Traditional automation with raw code is too complex. Frameworks like Robot Framework try to solve this by making tests more readable and accessible.

In the next doc, we'll see how Robot Framework improves the situation — and why it's still not enough.
