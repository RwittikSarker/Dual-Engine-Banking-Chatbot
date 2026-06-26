"""
responses.py
------------
Intent-to-response template mapping for all 77 Banking77 intents.

Since flan-t5-small is a very small model (80M params) it often produces
generic or question-style outputs. This module provides professional,
curated banking responses keyed by intent name, giving reliable and
meaningful answers to end users.

The RAG pipeline's Flan-T5 output is still used internally for the
generation score; these templates surface as the user-facing answer.
"""

from __future__ import annotations

INTENT_RESPONSES: dict[str, str] = {
    "Refund_not_showing_up": (
        "We're sorry to hear your refund hasn't appeared yet. "
        "Refunds typically take 3–5 business days to reflect on your account after being processed. "
        "If it's been longer, please contact our support team with your transaction reference and we will "
        "investigate and escalate this with the merchant on your behalf."
    ),
    "activate_my_card": (
        "You can activate your card quickly through the app: go to your card section, tap on your card, "
        "and select 'Activate Card'. You'll be prompted to confirm your PIN. "
        "If you don't have the app installed, you can also activate it via your first ATM withdrawal."
    ),
    "age_limit": (
        "Our accounts are available to customers aged 18 and above. "
        "For customers aged 16-17, we offer a Youth Account with parental consent required. "
        "Please visit our website or contact support for more information on eligibility."
    ),
    "apple_pay_or_google_pay": (
        "To set up Apple Pay or Google Pay, open your Wallet app (iOS) or Google Pay app (Android) "
        "and add your card by following the on-screen steps. "
        "You can also add it directly through the banking app under Card Settings > Mobile Payments. "
        "Contactless payments are enabled automatically once added."
    ),
    "atm_support": (
        "You can use your card at any Visa or Mastercard ATM worldwide. "
        "Standard withdrawal limits apply and fees may vary depending on the ATM operator. "
        "To find fee-free ATMs near you, use the ATM finder in our app."
    ),
    "automatic_top_up": (
        "Automatic top-ups can be set up in the app under Accounts > Top Up > Auto Top-Up. "
        "You can choose a minimum balance threshold and a fixed amount that will be automatically "
        "transferred from your linked funding source when your balance drops below that level."
    ),
    "balance_not_updated_after_bank_transfer": (
        "Bank transfers can take 1-3 business days to fully clear, depending on the sending bank. "
        "If you have a payment reference number, please check that the correct account details were used. "
        "If it's been more than 3 business days, please contact support with the transfer reference "
        "so we can trace the payment."
    ),
    "balance_not_updated_after_cheque_or_cash_deposit": (
        "Cheque and cash deposits may take up to 5 business days to clear. "
        "The funds will appear as pending in your transaction history during this period. "
        "If you believe there's a discrepancy after the clearing period, please contact our branch or support team."
    ),
    "beneficiary_not_allowed": (
        "Some beneficiaries may be blocked for security or compliance reasons. "
        "Please ensure the account details are correct and that the recipient is in a supported country. "
        "If you believe this is an error, please contact our support team with the transfer details "
        "for further review."
    ),
    "cancel_transfer": (
        "Transfers can be cancelled if they haven't yet been processed. "
        "Go to your transaction history, find the pending transfer, and tap 'Cancel Transfer'. "
        "If the transfer has already been sent, we may be able to recall it — please contact support immediately "
        "with the transaction reference."
    ),
    "card_about_to_expire": (
        "Your replacement card is automatically issued and should arrive within 7-10 business days "
        "of your current card's expiry. You can track delivery in the app under Card > Track Delivery. "
        "Your current card remains valid until the last day of the expiry month."
    ),
    "card_acceptance": (
        "Your card is accepted at all merchants that support Visa or Mastercard payments worldwide, "
        "both in-store and online. If a merchant doesn't accept your card, please check that the card "
        "is activated and that online/international payments are enabled in your card settings."
    ),
    "card_arrival": (
        "Your card should arrive within 5-7 business days of being ordered. "
        "You can track delivery status in real time through the app under Card > Track Delivery. "
        "If your card hasn't arrived after 10 business days, please report it as not received "
        "through the app so we can send a replacement."
    ),
    "card_delivery_estimate": (
        "Standard card delivery takes 5-7 business days. Express delivery (where available) takes 1-2 days. "
        "You can check the estimated delivery date and courier tracking in the app under Card > Track Delivery."
    ),
    "card_linking": (
        "To link your card to a third-party service (e.g. PayPal, subscription), simply use your card number, "
        "expiry date, and CVV displayed in the app. "
        "If linking to Apple Pay or Google Pay, use the in-app Card Settings > Mobile Payments option."
    ),
    "card_not_working": (
        "If your card isn't working, please check the following: "
        "(1) Ensure the card is activated in the app. "
        "(2) Check that the transaction type (contactless, chip, online) is enabled in Card Settings. "
        "(3) Verify you have sufficient funds. "
        "If all settings look correct, please contact support and we can issue a replacement card."
    ),
    "card_payment_fee_charged": (
        "Some merchants charge a card payment fee for card transactions, particularly for credit-style payments. "
        "This fee is charged by the merchant, not us. "
        "Please check the merchant's terms. If you believe this fee was charged in error, "
        "contact support with the transaction details so we can investigate."
    ),
    "card_payment_not_recognised": (
        "If you don't recognise a card payment, please check with family members or anyone who has access "
        "to your card. Merchant names on statements sometimes differ from the store name. "
        "If you still don't recognise it, please freeze your card immediately in the app "
        "and contact support to raise a dispute."
    ),
    "card_payment_wrong_exchange_rate": (
        "If you believe an incorrect exchange rate was applied, the rate used is determined at the time of "
        "transaction settlement, which may differ from the rate when the payment was initiated. "
        "Please contact support with the transaction reference and we'll investigate the rate applied."
    ),
    "card_swallowed": (
        "If an ATM has retained your card, please contact the ATM operator or bank immediately. "
        "In the meantime, freeze your card in our app under Card Settings > Freeze Card to prevent "
        "unauthorised use. Then contact our support team to arrange a replacement card."
    ),
    "cash_withdrawal_charge": (
        "Cash withdrawal fees depend on your account plan and the ATM operator. "
        "Our standard plan includes a set number of free ATM withdrawals per month, "
        "after which a small fee applies. Please check the Fees section in the app for your specific limits."
    ),
    "cash_withdrawal_not_recognised": (
        "If you don't recognise a cash withdrawal on your statement, please check with anyone who "
        "may have access to your card. If it is truly unrecognised, freeze your card immediately "
        "in the app and contact support to raise a dispute."
    ),
    "change_pin": (
        "You can change your PIN directly in the app: go to Card > Card Settings > Change PIN. "
        "You'll need to enter your current PIN to set a new one. "
        "Alternatively, you can change your PIN at any ATM that supports PIN services."
    ),
    "compromised_card": (
        "If you believe your card details have been compromised, freeze your card immediately "
        "in the app under Card > Freeze Card. Then contact our support team urgently to cancel the card "
        "and issue a replacement. We also recommend reviewing recent transactions for any unauthorised activity."
    ),
    "contactless_not_working": (
        "If contactless isn't working, try the following: "
        "(1) Check that contactless is enabled in Card Settings. "
        "(2) Ensure your card is held within 4 cm of the reader for 1-2 seconds. "
        "(3) Some contactless limits require a PIN — try inserting the card. "
        "If it still doesn't work, contact support and we can investigate."
    ),
    "country_support": (
        "Our card is accepted in most countries worldwide where Visa/Mastercard is supported. "
        "Some restricted countries may not be supported for compliance reasons. "
        "Please check the Country Support section in the app or contact support before travelling."
    ),
    "declined_card_payment": (
        "A declined payment can happen for several reasons: insufficient funds, the card not being activated, "
        "a daily spending limit being reached, or the merchant not accepting the card type. "
        "Check your balance and Card Settings in the app. If the issue persists, contact support for details."
    ),
    "declined_cash_withdrawal": (
        "A declined ATM withdrawal may be due to insufficient funds, your daily withdrawal limit being reached, "
        "or the ATM not accepting your card type. "
        "Please check your balance and account limits in the app. Contact support if the problem continues."
    ),
    "declined_transfer": (
        "A declined transfer may be due to incorrect account details, the recipient being in an unsupported "
        "country, a daily transfer limit, or a security hold on the account. "
        "Please double-check the recipient details and contact support if the issue persists."
    ),
    "direct_debit_payment_not_recognised": (
        "If you don't recognise a direct debit, it may be from a subscription or service you set up. "
        "Check your active direct debits in the app under Payments > Direct Debits. "
        "If you still don't recognise it, you can cancel it directly in the app and contact support "
        "to raise a dispute."
    ),
    "disposable_card_limits": (
        "Disposable virtual cards have a single-use limit by design — they expire after one transaction. "
        "For spending limits on disposable cards, please check your account plan in the app. "
        "You can generate a new disposable card in Card > Virtual Cards anytime."
    ),
    "edit_personal_details": (
        "You can update your personal details (name, address, phone number, email) in the app "
        "under Profile > Personal Details. Some changes may require identity verification. "
        "For legal name changes, please contact support and provide supporting documentation."
    ),
    "exchange_charge": (
        "Currency exchange charges depend on your account plan. Our standard plan offers a competitive "
        "exchange rate with no hidden fees for most currencies. A small fee may apply for certain exotic "
        "currencies. Check the Fees section in the app for your specific plan's exchange policy."
    ),
    "exchange_rate": (
        "We use the interbank exchange rate (or very close to it) for currency conversions. "
        "The exact rate applied is shown to you before you confirm a transaction. "
        "For live rates, check the Exchange section in the app."
    ),
    "exchange_via_app": (
        "To exchange currencies in the app: go to Accounts > Exchange, select the currencies you want "
        "to convert, enter the amount, review the rate, and confirm. "
        "The exchange is instant and funds will appear in your account immediately."
    ),
    "extra_charge_on_statement": (
        "If you see an unexpected charge on your statement, it may be a merchant fee, a currency conversion "
        "fee, or a service charge. Please check the transaction details in the app by tapping on the charge. "
        "If you believe it's unauthorised, contact support to raise a dispute."
    ),
    "failed_transfer": (
        "A failed transfer usually means the funds have been returned to your account. "
        "Common causes include incorrect account details or the recipient's bank rejecting the payment. "
        "Please verify the account details and try again. Contact support if funds haven't been returned "
        "within 3 business days."
    ),
    "fiat_currency_support": (
        "We support a wide range of fiat currencies for holding, exchange, and payments. "
        "To see the full list of supported currencies, go to Accounts > Currencies in the app. "
        "If a specific currency isn't listed, please contact support for more information."
    ),
    "get_disposable_virtual_card": (
        "To get a disposable virtual card, go to Card > Virtual Cards > Create Disposable Card in the app. "
        "These cards are single-use and expire after one transaction, making them ideal for online purchases "
        "where you don't want to use your main card details."
    ),
    "get_physical_card": (
        "To order a physical card, go to Card > Order Physical Card in the app. "
        "Enter your delivery address, confirm, and your card will arrive within 5-7 business days. "
        "Express delivery is available in select regions."
    ),
    "getting_spare_card": (
        "You can order a spare/additional card through the app under Card > Order Additional Card. "
        "Note that fees may apply for additional cards depending on your account plan. "
        "The spare card will be linked to the same account."
    ),
    "getting_virtual_card": (
        "To get a virtual card, go to Card > Virtual Cards > Create Virtual Card in the app. "
        "Your virtual card details (number, expiry, CVV) will be available instantly for online purchases. "
        "You can freeze, unfreeze, or delete virtual cards at any time."
    ),
    "lost_or_stolen_card": (
        "If your card is lost or stolen, please freeze it immediately through the app under "
        "Card > Freeze Card to prevent any unauthorised use. "
        "Then contact our support team to report it as lost/stolen and request a replacement card. "
        "If you suspect fraudulent transactions, we will raise a dispute on your behalf."
    ),
    "lost_or_stolen_phone": (
        "If your phone is lost or stolen, please contact us immediately so we can temporarily freeze "
        "your account access from that device. You can still log in from another device using your "
        "credentials. Change your app password as soon as possible via Profile > Security > Change Password."
    ),
    "order_physical_card": (
        "To order a physical card, open the app and go to Card > Order Physical Card. "
        "Confirm your delivery address and submit the request. "
        "Your card should arrive within 5-7 business days. Expedited delivery is available in select areas."
    ),
    "passcode_forgotten": (
        "If you've forgotten your passcode, tap 'Forgot Passcode' on the login screen. "
        "You'll be asked to verify your identity via email, phone number, or biometrics. "
        "Once verified, you can set a new passcode. If you can't access your verification methods, "
        "please contact support for manual identity verification."
    ),
    "pending_card_payment": (
        "Pending card payments are pre-authorised by the merchant and typically settle within 3-5 business days. "
        "During this time, the funds are temporarily held. Once the merchant processes the payment, "
        "the pending status will update to completed. Contact support if a pending charge hasn't cleared "
        "after 7 days."
    ),
    "pending_cash_withdrawal": (
        "ATM cash withdrawals are usually processed immediately, but in rare cases may show as pending "
        "for up to 24 hours. If a withdrawal remains pending after 24 hours, please contact support "
        "with the date, amount, and ATM location."
    ),
    "pending_top_up": (
        "Top-ups can take a few minutes to a few hours depending on the method used. "
        "Bank transfer top-ups may take 1-2 business days. Card top-ups are usually instant. "
        "If your top-up is still pending after 2 business days, please contact support with the "
        "payment confirmation reference."
    ),
    "pending_transfer": (
        "Transfers show as pending while they are being processed. Most transfers complete within "
        "1-3 business days. International transfers may take longer. "
        "If your transfer has been pending for more than 3 business days, please contact support "
        "with the transaction reference."
    ),
    "pin_blocked": (
        "Your PIN is blocked after multiple incorrect attempts as a security measure. "
        "To unblock it, you can reset your PIN via the app under Card > Card Settings > Unblock PIN. "
        "You may need to complete identity verification. Contact support if you need further assistance."
    ),
    "receiving_money": (
        "To receive money, share your account's sort code and account number (found in the app under "
        "Account Details). For international transfers, share your IBAN and SWIFT/BIC code. "
        "Incoming payments typically arrive within 1-3 business days."
    ),
    "request_refund": (
        "To request a refund for a transaction, first contact the merchant directly as they process "
        "refunds on their end. If the merchant is unresponsive or the charge is disputed, "
        "contact our support team with the transaction reference and we will raise a chargeback claim "
        "with Visa/Mastercard on your behalf."
    ),
    "reverted_card_payment?": (
        "A reverted card payment means the merchant has reversed a charge that was previously pending "
        "or completed. The funds should be returned to your account within 3-5 business days. "
        "If the funds haven't appeared after that period, please contact support with the transaction details."
    ),
    "supported_cards_and_currencies": (
        "We support Visa and Mastercard debit/credit cards for linking and payments. "
        "For currencies, we support a wide range — check the full list in Accounts > Currencies in the app. "
        "Most major global currencies are supported for holding and exchange."
    ),
    "terminate_account": (
        "We're sorry to see you go. To close your account, please go to Profile > Account Settings > "
        "Close Account in the app. Ensure your balance is zero before closing. "
        "If you have outstanding transactions or a remaining balance, transfer funds out first. "
        "Contact support if you need assistance with the closure process."
    ),
    "top_up_by_bank_transfer_charge": (
        "Topping up by bank transfer is free from most UK banks. "
        "Your sending bank may charge a fee for the outgoing transfer — please check with them. "
        "There are no charges on our end for receiving bank transfers."
    ),
    "top_up_by_card_charge": (
        "Topping up by debit card is free on standard plans. Some credit cards may incur a fee "
        "as the issuer may classify it as a cash advance. "
        "We recommend using a debit card or bank transfer for fee-free top-ups."
    ),
    "top_up_by_cash_or_cheque": (
        "We support cash top-ups at select retail locations (check the app for nearby partners). "
        "Cheque deposits can be made by posting to our registered address — processing takes up to 5 business days. "
        "For amounts over a certain limit, additional verification may be required."
    ),
    "top_up_failed": (
        "A failed top-up means your funds haven't been added to your account. "
        "Common causes include insufficient funds on the source card, incorrect card details, or "
        "a bank blocking the transaction. Please verify the details and try again. "
        "If funds were deducted from your source account, contact support immediately."
    ),
    "top_up_limits": (
        "Your daily, weekly, and monthly top-up limits depend on your account plan and verification level. "
        "To see your current limits, go to Account > Top Up > View Limits in the app. "
        "To increase your limits, you may need to complete additional identity verification."
    ),
    "top_up_reverted": (
        "A reverted top-up means the funds were initially credited but then reversed, usually because "
        "the source payment failed or was recalled. The funds should have been returned to your source account. "
        "Contact support if you can't see the reversal within 3-5 business days."
    ),
    "topping_up_by_card": (
        "To top up by card, go to Account > Top Up > Add Money by Card in the app. "
        "Enter the amount, select your card, and confirm. "
        "Card top-ups are instant and free with most debit cards."
    ),
    "transaction_charged_twice": (
        "We're sorry to hear you've been charged twice for the same transaction. "
        "Please contact our support team immediately with the transaction reference, amount, "
        "date, and merchant name. We will investigate the duplicate charge and initiate a "
        "refund for the incorrect charge within 3-5 business days. "
        "As a precaution, you may freeze your card while we investigate."
    ),
    "transfer_fee_charged": (
        "Transfer fees depend on the destination country, currency, and your account plan. "
        "Our international transfers are fee-free for standard currencies. "
        "Some exotic-currency transfers or SWIFT transfers may incur a fee, shown before you confirm. "
        "Check the Fees section in the app for your plan's specific transfer fee schedule."
    ),
    "transfer_into_account": (
        "To transfer funds into your account, simply share your account number and sort code with the sender. "
        "For international transfers, provide your IBAN and BIC/SWIFT code, found in the app under "
        "Account > Account Details. Incoming transfers typically arrive within 1-3 business days."
    ),
    "transfer_not_received_by_recipient": (
        "If the recipient hasn't received a transfer, it may still be processing. "
        "Bank transfers can take 1-3 business days. International transfers may take longer. "
        "Please provide the recipient with the exact transfer reference number to check with their bank. "
        "Contact our support team if the transfer hasn't arrived after 5 business days."
    ),
    "transfer_timing": (
        "Domestic transfers usually arrive within the same business day. "
        "International transfers typically take 1-5 business days depending on the destination country "
        "and the recipient's bank processing times. "
        "You can track your transfer status in real time in the app under Transactions."
    ),
    "unable_to_verify_identity": (
        "Identity verification failures can occur due to blurry document photos, expired documents, "
        "or a name mismatch between your ID and account. "
        "Please ensure your document is valid, the photo is clear, and your name matches exactly. "
        "Contact support if you continue to have difficulties and an agent will assist you manually."
    ),
    "verify_my_identity": (
        "To verify your identity, go to Profile > Identity Verification in the app. "
        "You'll be asked to upload a valid government-issued photo ID (passport or driving licence) "
        "and a selfie. Verification is usually completed within a few minutes to 24 hours. "
        "Ensure your documents are clear, valid, and unexpired."
    ),
    "verify_source_of_funds": (
        "We may request source of funds verification to comply with anti-money laundering (AML) regulations. "
        "You'll be asked to provide documentation such as payslips, bank statements, or employment letters. "
        "Please upload the requested documents through the app or as instructed by our support team."
    ),
    "verify_top_up": (
        "Top-up verification is required for larger deposits to comply with financial regulations. "
        "You'll be asked to confirm the source of the funds (e.g., bank statement, payslip). "
        "Please submit the requested documents through the in-app verification flow or by contacting support."
    ),
    "virtual_card_not_working": (
        "If your virtual card isn't working for online payments, please check that: "
        "(1) The card is not frozen. (2) Online payments are enabled in Card Settings. "
        "(3) You're entering the correct card number, expiry, and CVV shown in the app. "
        "Some merchants don't accept virtual cards — try a different merchant or contact support."
    ),
    "visa_or_mastercard": (
        "Your card network (Visa or Mastercard) is shown on your card and in the app. "
        "Both are accepted globally at millions of merchants. The specific network you have "
        "depends on the card issued when you joined. Contact support if you need a card on a "
        "specific network."
    ),
    "why_verify_identity": (
        "Identity verification is required by law under Know Your Customer (KYC) and Anti-Money "
        "Laundering (AML) regulations. This protects both you and us from fraud and financial crime. "
        "All banks and financial institutions are legally required to verify customer identities "
        "before providing full account access."
    ),
    "wrong_amount_of_cash_received": (
        "If you received an incorrect amount from an ATM, please take note of the ATM location, "
        "time, and amount displayed versus dispensed. Contact our support team immediately with this "
        "information. We will raise a dispute with the ATM operator on your behalf. "
        "Keep any receipts as evidence."
    ),
    "wrong_exchange_rate_for_cash_withdrawal": (
        "The exchange rate applied to a cash withdrawal is determined by the ATM operator or your card "
        "scheme at the time of transaction. You may have been offered Dynamic Currency Conversion (DCC) — "
        "always choose to pay in the local currency to get the best rate. "
        "Contact support with the withdrawal details if you believe the rate was incorrect."
    ),
}

_FALLBACK_RESPONSE = (
    "Thank you for reaching out to our banking support. "
    "I've noted your query and a specialist will review it shortly. "
    "For urgent matters, please contact our 24/7 support line directly. "
    "We apologise for any inconvenience caused."
)


def get_response(intent: str) -> str:
    """Return a professional response for the given Banking77 intent."""
    return INTENT_RESPONSES.get(intent, _FALLBACK_RESPONSE)

